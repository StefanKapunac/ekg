from matplotlib import pyplot as plt
import cv2
import numpy as np
import statistics
import pytesseract
from pytesseract import Output
import re
import string
import json

VERBOSE = True

def display_image(image, caption='Image', grayscale=True):
    if not VERBOSE:
        return

    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(image.shape[1] / 300.0, image.shape[0] / 300.0))
    # fig = plt.figure(figsize=(image.shape[1] / 150.0, image.shape[0] / 150.0))
    # ax = fig.add_subplot(111)
    ax = plt.axes([0,0,1,1])
    if grayscale:
        ax.imshow(img2, interpolation='none', cmap = 'gray')
    else:
        ax.imshow(img2, interpolation='none')
    # ax.set_title(caption)
    plt.axis('off')
    # if caption == 'squares':
    	# fig.savefig('results/' + caption + '.png', dpi=300, transparent=True)
    # fig.savefig('za_pdf/' + caption + '.png', dpi=150, transparent=True)
    plt.show()


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

def canny(image):
    return cv2.Canny(image, 100, 200)

# mozda cak i radi nesto kao
def find_lines(color_img, edges):
    img_cpy = color_img.copy()
    # 0.1 degree accuracy, 70% of width
    # eventualno napraviti petlju, ako je lines None,
    # da proba sa manjim procentom od 70 zbog slika gde se mreza slabo vidi
    # npr.
    percents = [0.9, 0.7, 0.5, 0.3]
    for percent in percents:
        lines = cv2.HoughLines(edges, 1, np.pi/3600, int(percent * color_img.shape[1]))
        if lines is None: # mozda i neko ogranicenje tipa < 5
            continue
        print(percent, len(lines))
        angles = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = x0 + img_cpy.shape[1]*(-b)
            y1 = y0 + img_cpy.shape[1]*(a)
            x2 = x0 - img_cpy.shape[1]*(-b)
            y2 = y0 - img_cpy.shape[1]*(a)

    #         m = (y2 - y1) / (x2 - x1)
    #         angle = np.degrees(np.arctan(m))
    #         print(np.degrees(theta), angle)
            angle = np.degrees(theta) - 90
            angles.append(angle)


            cv2.line(img_cpy,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)
            
        break

    display_image(img_cpy, 'lines', grayscale=False)
    
    print(angles)
    try:
        result = statistics.mode(angles)
    except statistics.StatisticsError:
        result = statistics.median(angles)
    
    return result

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# treba da se popravi...
def find_squares(image, edges):
    image_copy = image.copy()
    # num_squares = 0
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(gray, contours, -1, (0,255,0))
    # display_image(gray, grayscale=False)
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)


        # four vertices
        if len(approx) == 4 and cv2.contourArea(cnt) > 10 and cv2.isContourConvex(cnt):
            rect = cv2.minAreaRect(approx)
            print(rect)
            (x, y), (width, height), angle = rect
            if width != height:
                print('not a square')
                continue
            
            angle = rect[-1]
            box = cv2.boxPoints(rect)
            print(box)
            box = np.int0(box)
            cv2.drawContours(image_copy, [box], 0, (255, 0, 0), 2)

    display_image(image_copy, 'squares', grayscale=False)

def get_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def normalize(image):
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def extract_background(image):
    hsv_3 = get_hsv(image)
    v = hsv_3[:,:,2]
#     plt.hist(v)
    mask = cv2.bitwise_not(v)
    display_image(mask, 'Maska')

    gray = get_grayscale(image)
    inverted_gray = cv2.bitwise_not(gray)
    display_image(inverted_gray, 'Grayscale')
    
    background = inverted_gray - mask
    return background

def ocr_words(image):
    image_copy = image.copy()
    d = pytesseract.image_to_data(image_copy, output_type=Output.DICT)
    print(d.keys())
    
    # TODO: mozda je bolje umesto regularnog izraza samo provera da li niska sadrzi mm/s tj. mm/mV
    mm_s_pattern = '^\d\dmm/s$'
    mm_v_pattern = '^\d\dmm/mV$'
    
    mm_s = None
    mm_v = None
    
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            
            # desi se da uzme deo signala jer misli da je | ili /...
            if d['text'][i].isspace() or d['text'][i].strip() in string.punctuation:
                continue
            
            
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
            
            print(d['text'][i])
            if re.match(mm_s_pattern, d['text'][i]):
                print('milimetri po sekundi')
                mm_s = int(d['text'][i][:2])
                
            elif re.match(mm_v_pattern, d['text'][i]):
                print('milimetri po milivoltu')
            
    display_image(image_copy, 'ocr')
    print('{} milimetara po sekundi'.format(mm_s))
#     print('najveci pravougaonik: - {} -'.format(d['text'][max_width_ind]))

    return image_copy

def find_leads(img, original_img):
	# izbroj bele piksele u svakom redu
	nonzeros = np.count_nonzero(img, axis=1)

	if VERBOSE:
		plt.plot(nonzeros)
		plt.show()

	# za svaki red najveci crni region
	black_regions = []
	for i in range(img.shape[0]):
	    biggest = 0
	    current = 0
	    continue_region = True
	    for j in range(img.shape[1]):
	        if continue_region and img[i,j] == 0:
	            current += 1
	        elif continue_region:
	            if current > biggest:
	                biggest = current
	                
	            continue_region = False
	            current = 0
	        elif img[i,j] == 0:
	            continue_region = True
	            current += 1
	            
	    if current > biggest:
	        biggest = current
	    black_regions.append(biggest)

	# plt.plot(black_regions)
	print(np.argmin(black_regions))
	            
	weighted = nonzeros / black_regions
	    

	max_peak = np.max(weighted)
	signal_zeros = []
	signal_zeros_ind = []
	for i in range(len(weighted)):
	    if len(signal_zeros) == 0 and weighted[i] > 0.3 * max_peak and i > int(0.01*len(weighted)) and i < int(0.99*len(weighted)) and np.all(weighted[i] >= np.array(weighted[int(i - 0.01*len(weighted)):int(i + 0.01*len(weighted))])):
	        signal_zeros.append(weighted[i])
	        signal_zeros_ind.append(i)
	    
	    elif len(signal_zeros) > 0 and weighted[i] > 0.3 * max_peak and i - signal_zeros_ind[-1] > 0.04 * len(weighted) and i > int(0.01*len(weighted)) and i < int(0.99*len(weighted)) and np.all(weighted[i] >= np.array(weighted[int(i - 0.01*len(weighted)):int(i + 0.01*len(weighted))])):
	        signal_zeros.append(weighted[i])
	        signal_zeros_ind.append(i)
	        
	    
	        print(i)
	        print(weighted[i])
	        print(weighted[int(i - 0.01*len(weighted)):int(i + 0.01*len(weighted))])


	if VERBOSE:
		plt.plot(weighted)	        
		plt.scatter(signal_zeros_ind, signal_zeros, c='red')
		plt.show()

	# nacrtane linije na slici
	signal_lines_img = original_img.copy()
	for zero in signal_zeros_ind:
	    cv2.line(signal_lines_img, (0, zero), (img.shape[1], zero), (0, 255, 0), 5)

	display_image(signal_lines_img, 'signal lines')

	return signal_zeros_ind

def find_lead_start(img, signal_zeros_ind):
	image_dots = img.copy()
	width = img.shape[1]
	height = img.shape[0]
	start_points = []
	for zero in signal_zeros_ind:
	    # ne krece se bas od prve kolone, tu je moguce da postoji neki okvir ili tako nesto...
	    offset = int(0.05 * width)
	    for j in range(offset, width):
	        region_start = None
	        region_end = None
	        eps = int(0.01 * height)
	        for i in range(zero - eps, zero + eps):
	#             print(i, j)
	#             print(img[i,j])
	            if img[i,j] != 0 and region_start is None:
	                region_start = i
	                region_end = i
	            elif img[i,j] != 0:
	                region_end = i
	            elif region_start is not None:
	                break
	        
	        if region_start is not None:
	            start_points.append((j, (region_start + region_end) // 2))
	            break
	#     break
	    cv2.circle(image_dots, start_points[-1], 15, (255,255,255),5)
	#     cv2.line(image_dots, (0, zero), (width, zero), (0, 255, 0), 5)
	    
	display_image(image_dots, 'start of signals')

	# popravka - uzmi medijanu x koordinata kao konacno x za sve leadove
	xs = [x for (x,y) in start_points]
	median_x = statistics.median(xs)
	print(median_x)

	image_copy = img.copy()
	cv2.line(image_copy, (int(median_x), 0), (int(median_x), image_copy.shape[0]), (0,255,0), 5)
	display_image(image_copy, 'start of signals - median')

	return median_x

def extract_signal_5(img, signal_zeros_ind, original_img, median_x):
	x_start = int(median_x)
	signal_ys = [[y_start] for y_start in signal_zeros_ind]
	signal_xs = [[x_start] for y_start in signal_zeros_ind]
	print(signal_ys)
	height = img.shape[0]
	width = img.shape[1]

	min_lead_distance = signal_zeros_ind[1] - signal_zeros_ind[0]
	for i in range(1, len(signal_zeros_ind) - 1):
		current_lead_distance = signal_zeros_ind[i + 1] - signal_zeros_ind[i]
		if current_lead_distance < min_lead_distance:
			min_lead_distance = current_lead_distance
	print('min_lead_distance: ', min_lead_distance)
	max_radius = min_lead_distance // 2

	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

	# Map component labels to hue val, 0-179 is the hue range in OpenCV
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
	labeled_img[label_hue==0] = 0

	display_image(labeled_img)

	print(num_labels)
	print(labels)
	print(stats)
	print(type(labels))

	# za svaki kanal nadji najvecu komponentu koja je u redu gde je nula
	# onda je produzi na levo i na desno dokle treba tako sto je spajas sa najblizom komponentom

	biggest_components = [None for s in signal_ys]
	for k in range(len(signal_ys)):
		# nadji najvecu komponentu u redu
		labels_in_row = labels[signal_ys[k],:][0]
		# print(type(labels_in_row))
		
		values, counts = np.unique(labels_in_row[labels_in_row != 0], return_counts=True)
		ind = np.argmax(counts)
		biggest_components[k] = values[ind]

	print(biggest_components)


	# TODO: ubrzaj - koristi numpy...
	# labeled_img_biggest = labeled_img.copy()
	# for i in range(height):
	# 	for j in range(width):
	# 		if labels[i, j] not in biggest_components:
	# 			labeled_img_biggest[i, j] = 0
	# display_image(labeled_img_biggest)

	nema = 0
	# radius = height // 100
	# radius = 20
	radius = 10
	neighbourhoods = [(s[-1] - radius, s[-1] + radius) for s in signal_ys]
	prev_label = biggest_components.copy()
	# za svaku kolonu
	for j in range(x_start + 1, width):
		closest_regions = [None for lead in range(len(signal_ys))]
		closest_labels = [None for lead in range(len(signal_ys))]
		# za svaki kanal
		for k in range(len(signal_ys)):
			if prev_label[k] != 0:
				# vidi da li i u ovoj koloni postoji ta komponenta
				# ako postoji samo nastavi sa njom
				closest_region = None
				region_start = None
				region_end = None
				same_label = False
				# for i in range(signal_ys[k][-1] - radius * 5, signal_ys[k][-1] + radius * 5 + 1):
				for i in range(signal_ys[k][-1] - radius * 10, signal_ys[k][-1] + radius * 10 + 1):
					if i < 0:
						i = -1
						continue
					elif i >= height:
						break


					if region_start is None and labels[i, j] == prev_label[k]:
						region_start = i
						region_end = i
					elif labels[i, j] == prev_label[k]:
						region_end = i
					elif region_start is not None:
						region_mean = (region_start + region_end) // 2
						if closest_region is None or abs(signal_ys[k][-1] - region_mean) < abs(signal_ys[k][-1] - closest_region):
							closest_region = region_mean

						# i prev label ostaje isti
						same_label = True
				if same_label:
					closest_regions[k] = closest_region
					closest_labels[k] = prev_label[k]

				if not same_label:
					closest_region = None
					found_label = None
					closest_label = None
					if neighbourhoods[k][1] - neighbourhoods[k][0] > max_radius:
						neighbourhoods[k] = signal_ys[k][-1] - max_radius // 2, signal_ys[k][-1] + max_radius // 2
					for i in range(neighbourhoods[k][0], neighbourhoods[k][1] + 1):
						if i < 0:
							i = -1
							continue
						elif i >= height:
							break

						if region_start is None and labels[i, j] != 0:
							region_start = i
							region_end = i
							found_label = labels[i, j]
						elif region_start is not None and labels[i, j] == found_label:
							region_end = i
						elif region_start is not None and labels[i, j] != found_label:
							region_mean = (region_start + region_end) // 2
							# ako je to prvi ili najblizi region
							if closest_region is None or abs(signal_ys[k][-1] - region_mean) < abs(signal_ys[k][-1] - closest_region):
								closest_region = region_mean						 
								closest_label = found_label

							region_start = None
							region_end = None
							found_label = None

					if closest_region is None:
						# print('nema: ', i, j)
						nema+=1
						neighbourhoods[k] = neighbourhoods[k][0] - radius // 10, neighbourhoods[k][1] + radius // 10
						
						# print(k, j, neighbourhoods[k])
					else:
						# ako postoji neki drugi kanal koji je blizi, onda ne moze
						closest_regions[k] = closest_region
						closest_labels[k] = closest_label


		chosen_regions = [None for c in closest_regions]
		chosen_labels = [None for c in closest_regions]
		for k1 in range(len(closest_regions)):
			if closest_regions[k1] is None:
				continue
			if closest_labels[k1] == prev_label[k1]:
				chosen_regions[k1] = closest_regions[k1]
				chosen_labels[k1] = prev_label[k1]
			else:
				colision = False
				for k2 in [k1-1, k1+1]:
					if k2 < 0 or k2 >= len(closest_regions):
						continue
					if closest_regions[k2] is not None and closest_labels[k1] == closest_labels[k2] and abs(closest_regions[k1] - closest_regions[k2]) < 10:
						colision = True
						print(j, k1, k2, closest_regions[k1], closest_regions[k2])
						if closest_labels[k2] == prev_label[k2]:
							continue
						dist_1 = abs(closest_regions[k1] - signal_ys[k1][-1])
						dist_2 = abs(closest_regions[k2] - signal_ys[k2][-1])
						if dist_1 < dist_2:
							chosen_regions[k1] = closest_regions[k1]
							chosen_labels[k1] = closest_labels[k1]
					# if closest_regions[k2] is not None and abs(closest_regions[k1] - closest_regions[k2]) < 20:
				# 		colision = True
				# 		print(j, k1, k2, closest_regions[k1], closest_regions[k2])
				# 		if closest_labels[k2] == prev_label[k2]:
				# 			continue
				# 		dist_1 = abs(closest_regions[k1] - signal_ys[k1][-1])
				# 		dist_2 = abs(closest_regions[k2] - signal_ys[k2][-1])
				# 		if dist_1 < dist_2:
				# 			chosen_regions[k1] = closest_regions[k1]
				# 			chosen_labels[k1] = closest_labels[k1]
				if not colision:
					chosen_regions[k1] = closest_regions[k1]
					chosen_labels[k1] = closest_labels[k1] 
				else:
					neighbourhoods[k1] = neighbourhoods[k1][0] - radius // 2, neighbourhoods[k1][1] + radius // 2

		# print(chosen_regions)
		# print(chosen_labels)
		# print(prev_label)
		for cr_i in range(len(chosen_regions)):
			if chosen_regions[cr_i] is None:
				# neighbourhoods[cr_i] = neighbourhoods[cr_i][0] - radius // 10, neighbourhoods[cr_i][1] + radius // 10
				neighbourhoods[cr_i] = neighbourhoods[cr_i][0] - radius // 2, neighbourhoods[cr_i][1] + radius // 2
			else:
				signal_ys[cr_i].append(chosen_regions[cr_i])
				signal_xs[cr_i].append(j)
				prev_label[cr_i] = chosen_labels[cr_i]
				neighbourhoods[cr_i] = signal_ys[cr_i][-1] - radius, signal_ys[cr_i][-1] + radius

			# u prehodnoj koloni je bila rupa
			# if prev_label[k] == 0:

	print([len(x) for x in signal_xs])
	print([len(y) for y in signal_ys])
	print('ukupno nema: ', nema)

	raw_signal_image = original_img.copy()	
	for l in range(len(signal_ys)):
	    # print(len(signal_ys[l]), len(signal_xs[l]))

	    points = np.array([list(e) for e in zip(signal_xs[l], signal_ys[l])])
	    cv2.polylines(raw_signal_image, [points], isClosed=False, color=(0,255,0), thickness=5)

	display_image(raw_signal_image, 'raw signal')

	# interpolacija
	final_xs = range(x_start, width)
	final_ys = [[None for x in final_xs] for i in signal_ys]
	# za svaki kanal
	for k in range(len(signal_xs)):
		j = 0
		for i in final_xs:
			# print(k, j)
			if i == signal_xs[k][j]:
				final_ys[k][i - x_start] = signal_ys[k][j]
				j += 1
			else:
				slope = (signal_ys[k][j] - signal_ys[k][j-1]) / (signal_xs[k][j] - signal_xs[k][j-1])
				final_ys[k][i - x_start] = int(signal_ys[k][j-1] + slope * (i - signal_xs[k][j-1]))


	return_xs = np.array(final_xs)
	return_xs -= x_start

	return_ys = np.array(final_ys)
	for i in range(len(final_ys)):
		return_ys[i] -= signal_zeros_ind[i]

	for i in range(len(final_ys)):
		plt.plot(return_xs, return_ys[i])
	plt.show()

	return raw_signal_image, return_xs, return_ys


def digitize(filename):
	img = cv2.imread(filename)
	display_image(img, 'original')
	
	gray = get_grayscale(img)
	display_image(gray, 'grayscale')

	edges = canny(gray)
	display_image(edges, 'edges')

	angle = -0.1500091552734375
	# angle = -0.3000030517578125
	# angle = find_lines(img, edges)
	print(angle)
	rotated = rotate(img, angle)
	display_image(rotated, 'rotated')
	gray_rotated = get_grayscale(rotated)

	thresh = thresholding(gray_rotated)
	display_image(thresh, 'thresholded')

	edges_rotated = canny(gray_rotated)
	display_image(edges_rotated, 'edges rotated')
	# find_squares(rotated, edges_rotated)

	background = extract_background(rotated)
	display_image(background, 'background')

	cleared = gray_rotated + background
	display_image(cleared, 'background cleared')

	cleared_text = ocr_words(cleared)
	
	thresh_text = thresholding(cleared_text)
	display_image(thresh_text, 'cleared text binarized')

	thresh_text_inv = cv2.bitwise_not(thresh_text)
	display_image(thresh_text_inv, 'inverted')

	signal_zeros_ind = find_leads(thresh_text_inv, rotated)

	median_x = find_lead_start(thresh_text_inv, signal_zeros_ind)

	# extract_signal(thresh_text_inv, signal_zeros_ind, rotated, median_x)

	# dilated = cv2.dilate(thresh_text_inv, (5,5), iterations=10)
	dilated = cv2.dilate(thresh_text_inv, (5,5), iterations=5)
	display_image(dilated, 'dilated')
	


	raw_signal_image, final_xs, final_ys = extract_signal_5(dilated, signal_zeros_ind, rotated, median_x)

	return raw_signal_image, final_xs, final_ys

def convert_to_time_voltage(xs, ys, filename):
	num_leads = len(ys)

	# podeli xs i ys na 12 kanala
	leads_xs = None
	leads_ys = None
	if num_leads == 12:
		leads_xs = xs
		leads_ys = ys
	elif num_leads == 6:
		leads_xs = range(len(xs) // 2)
		leads_ys = [None for i in range(12)]
		for i in range(6):
			leads_ys[i] = ys[: len(ys) // 2]
			leads_ys[i + 6] = ys[len(ys) // 2 : ]
	elif num_leads == 3:
		leads_xs = range(len(xs) // 4)
		leads_ys = [None for i in range(12)]
		for i in range(3):
			quarter = len(ys) // 4
			leads_ys[i] = ys[: quarter]
			leads_ys[i + 3] = ys[quarter : 2 * quarter]
			leads_ys[i + 6] = ys[2 * quarter : 3 * quarter]
			leads_ys[i + 9] = ys[3 * quarter : ]
	else:
		print('pogresan broj kanala...')

	lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
	digitized = {lead : None for lead in lead_names}

	MM_IN_INCH = 25.4
	# TODO: ovde dodaj ono opciju kao argument da se unosi
	# ili foru na osnovu onih nadjenih kvadrata da se izracuna
	RESOLUTION = 300
	pixel_size = MM_IN_INCH / RESOLUTION

	# TODO: dodaj ono sto je nasao ocr
	SPEED = 25
	VOLTAGE = 10

	pixel_time = pixel_size / SPEED
	pixel_voltage = pixel_size / VOLTAGE

	print(f'pixel size: {pixel_size}')
	print(f'pixel time: {pixel_time}')
	print(f'pixel voltage: {pixel_voltage}')

	for i in range(len(lead_names)):
		time = [x * pixel_time for x in leads_xs]
		voltage = [y * pixel_voltage for y in leads_ys[i]]
		digitized[lead_names[i]] = list(zip(time, voltage))


	with open(f'{filename}.json', 'w') as f:
		json.dump(digitized, f, indent=4)

def main():
	raw_signal_image, final_xs, final_ys = digitize('images/9.jpg')
	# raw_signal_image, final_xs, final_ys = digitize('images/ecg-sample.jpg')

	convert_to_time_voltage(final_xs, final_ys, 'digitized/9')




if __name__ == '__main__':
	main()