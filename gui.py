import PySimpleGUI as sg
import os.path
from PIL import Image
from io import BytesIO
import cv2
import ekg

IMG_WIDTH = 750
IMG_HEIGHT = 450

def get_img_data(f, first=True):
    """Generate image data using PIL
    """
    img = Image.open(f)
    print(type(img))
    cur_width, cur_height = img.size
    print(img.size)
    new_width, new_height = IMG_WIDTH, IMG_HEIGHT
    scale_w = new_width / cur_width
    scale_h = new_height / cur_height 
    img = img.resize((int(cur_width*scale_w), int(cur_height*scale_h)), Image.ANTIALIAS)
    print(img.size)
    # img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

layout = [
	[sg.FileBrowse('Choose image', target='-FILE-'), sg.Input(enable_events=True, key='-FILE-', visible=False)],
	[sg.Image(key='-IMAGE-', size=(800, 450))],
	[sg.Text('Resolution (dpi)', key='-RES TEXT-', size=(17,1)), sg.Input(size=(5,1), key='-RESOLUTION-', disabled=True), sg.Text(size=(23,1))],
	[sg.Text('Speed (mm/s)', key='-SPEED TEXT-', size=(17,1)), sg.Input(size=(5,1), key='-SPEED-', disabled=True), sg.Text(size=(23,1))],
	[sg.Text('Voltage (mm/mV)', key='-VOLTAGE TEXT-', size=(17,1)), sg.Input(size=(5,1), key='-VOLTAGE-', disabled=True), sg.Text(size=(23,1))],
	[sg.Text('Output file name', key='-OUTPUT TEXT-', size=(17, 1)), sg.Input(size=(17,1), key='-OUTPUT-', disabled=True, enable_events=True), sg.Text('', size=(23,1), key='-ERRMSG-', text_color='red')],
	[sg.Button('Digitize', key='-DIGITIZE-', disabled=True)]
]

disabled_elems = ['-RESOLUTION-', '-SPEED-', '-VOLTAGE-', '-DIGITIZE-', '-OUTPUT-']

window = sg.Window('ECG digitizer', layout, size=(800,600), element_justification='center')

while True:
	event, values = window.read()
	if event == sg.WIN_CLOSED:
		break
	elif event == '-FILE-':
		filename = values['-FILE-']
		if not filename.endswith(('.png', '.jpg')):
			continue
		window['-IMAGE-'].update(data=get_img_data(filename, first=True))
		for i in disabled_elems:
			print(i)
			window[i].update(disabled=False)
	elif event == '-OUTPUT-':
		if values['-OUTPUT-'].strip() != '':
			window['-ERRMSG-'].update('')
		else:
			window['-ERRMSG-'].update('This field is necessary')

	elif event == '-DIGITIZE-':
		print('digitalizuj')
		if values['-OUTPUT-'].strip() == '':
			window['-ERRMSG-'].update('This field is necessary')
			continue

		res = values['-RESOLUTION-']
		speed = values['-SPEED-']
		voltage = values['-VOLTAGE-']
		print('fajl: ' + values['-FILE-'])
		result = ekg.digitize(values['-FILE-'])[0]
		resized = cv2.resize(result, (IMG_WIDTH, IMG_HEIGHT))
		print(resized)
		imgbytes = cv2.imencode(".png", resized)[1].tobytes()
		window['-IMAGE-'].update(data=imgbytes)


window.close()