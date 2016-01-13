#
# Text image generator
#
# usage: python dataset_generator.py test_input.jpg "A.G.Polat"
# Create text images with various fonts according to various criteria
# AGP Nov '15

import string
from subprocess import Popen, PIPE

from SimpleCV import *

from util import misc


def add_noise(img, percent=0.1):
    narr = img.getNumpy()
    amount = int((img.width * img.height) * percent)
    for i in range(amount):
        r = random.randrange(0, 255)
        color = (r, r, r)
        narr[random.randrange(img.width)][random.randrange(img.height)] = color
    return Image(narr)


def above_the_line(x1, y1, x2, y2, px, py):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


def find_xy(width, height):
    return width * random.random(), height * random.random()


def find_x2y2(width, height, x1):
    return x1 + (width - x1) * random.random(), height * random.random()


def scale(xs, ys, width, height):
    wmax = max(xs)
    wmin = min(xs)
    hmax = max(ys)
    hmin = min(ys)
    sx = float(width) * 0.9 / (wmax - wmin)
    sy = float(height) * 0.9 / (hmax - hmin)
    mx = float(wmax + wmin) / 2.0
    my = float(hmax + hmin) / 2.0
    for i in range(0, len(xs)):
        xs[i] -= mx
    for i in range(0, len(xs)):
        xs[i] *= sx
    for i in range(0, len(ys)):
        ys[i] -= my
    for i in range(0, len(ys)):
        ys[i] *= sy
    for i in range(0, len(ys)):
        ys[i] = min(height * (0.6 + random.random() * 0.8), ys[i] + height * (0.4 + random.random() * 0.25))
    for i in range(0, len(xs)):
        xs[i] = min(width * (0.6 + random.random() * 0.8), xs[i] + width * (0.4 + random.random() * 0.25))
    # print(zip(xs,ys))
    return [(x, y) for (x, y) in zip(xs, ys)]


def clockwise_quadrilateral_old(width, height):
    """return a random list of 4 different points in the clockwise direction"""
    # upper left corner
    (x1, y1) = find_xy(width / 2, height / 2)
    # upper right corner
    (x2, y2) = find_x2y2(width, height, x1)
    # select from underneath of the line
    # lower right corner
    (x3, y3) = find_xy(width, height)
    i = 0
    while above_the_line(x1, y1, x2, y2, x3, y3) >= 0:
        i += 1
        if i > 5:  # reassign
            (x1, y1) = find_xy(width / 2, height / 2)
            (x2, y2) = find_x2y2(width, height, x1)
            i = 0
        (x3, y3) = find_xy(width, height)
    # lower left corner
    (x4, y4) = find_xy(width, height)
    i = 0
    while (above_the_line(x1, y1, x2, y2, x4, y4) >= 0 or
                   above_the_line(x1, y1, x3, y3, x4, y4) >= 0 or
                   above_the_line(x2, y2, x3, y3, x4, y4) >= 0):
        i += 1
        if i > 30:
            (x1, y1) = find_xy(width / 2, height / 2)
            (x2, y2) = find_x2y2(width, height, x1)
            (x3, y3) = find_xy(width, height)
            i = 0
        (x4, y4) = find_xy(width, height)
    return scale([x1, x2, x3, x4], [y1, y2, y3, y4], width, height)


def clockwise_quadrilateral(width, height):
    xs = [random.randint(0, int(width * (35.0 / 200))) for i in range(0, 4)]
    xs[1] += int(width * (165.0 / 200))  # skip extremely warped texts
    xs[2] += int(width * (165.0 / 200))
    ys = [random.randint(0, int(height * (35.0 / 200))) for i in range(0, 4)]
    ys[2] += int(height * (165.0 / 200))
    ys[3] += int(height * (165.0 / 200))
    return scale(xs, ys, width, height)  # [(x,y) for (x,y) in zip(xs,ys)]#


def generate_letter_images():
    """generate a new dataset of warped characters [a-zA-Z0-9]"""
    if "(" in sys.argv[1]:  # assume a dimension input and an empty bg
        [w, h] = sys.argv[1][1:-1:].split(',')
        print(int(w) * int(h))
    with open("/home/agp/workspace/deep_learning/streetView/new_dataset/train_with_inverted.csv", "w") as train_csv:
        train_csv.write("label," + ",".join(["pixel" + str(i) for i in range(0, int(w) * int(h))]) + "\n")
        for text in string.ascii_letters + string.digits:  # for all alphanumeric
            print(text)
            if "(" in sys.argv[1]:  # assume a dimension input and an empty bg
                img = Image(tuple([int(i) for i in sys.argv[1][1:-1:].split(',')]))
                fname = "test_input.Bmp"
            else:
                img = Image(sys.argv[1]).invert()  # background e.g. Image((150,100))#
                fname = sys.argv[1]
            path = "/home/agp/workspace/deep_learning/streetView/new_dataset/"
            sys.stdout.flush()
            # text = sys.argv[2] # e.g. "A.G.Polat"
            w = img.width
            h = img.height

            # extended fonts: wont work in windows!
            if False and "Linux" in platform.system():  # skip them for now
                fonts = Popen('find /usr/share/fonts | grep ttf', shell=True, stdin=PIPE,
                              stdout=PIPE).stdout.read().strip().split('\n') + img.dl().listFonts()
            else:
                fonts = img.dl().listFonts()
            # print(fonts)
            # fonts=xfonts+fonts
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                      range(0, len(fonts))]
            i = 0
            newImg = img.copy()
            img2 = Image(img.size())
            for font in fonts:
                if font == "" or font in ["qbicle1brk", "dblayer1brk", "dblayer3brk", "droidsanshebrew", "mrykacstqurn",
                                          "zoetropebrk", "lohitpunjabi", "loopybrk", "kacstqurn", "droidsansgeorgian",
                                          "mallige", "webdings", "3dletbrk", "scalelinesmazebrk", "kacstpen",
                                          "nucleusbrk", "unresponsivebrk", "kacstart", "kacstdecorative", "dblayer2brk",
                                          "kacstoffice", "binaryxbrk", "binaryx01sbrk", "lklug", "symmetrybrk",
                                          "linedingsbrk", "droidsansarmenian", "zurklezoutlinebrk",
                                          "entangledlayerbbrk", "kacstdigital", "skullcapzbrk", "kacstbook",
                                          "konectoro1brk", "yourcomplexobrk", "opensymbol", "18holesbrk",
                                          "droidsansjapanese", "qbicle4brk",
                                          "qbicle3brk", "lohitdevanagari", "doublebogeybrk", "binary01sbrk",
                                          "kacstscreen", "lohitbengali", "kacstposter", "kedage", "kacstone",
                                          "kacstnaskh", "droidsansethiopic", "qbicle2brk", "vemana2000", "kacstfarsi",
                                          "xmaslightsbrk", "headdingmakerbrk", "kacstletter", "binarybrk", "ori1uni",
                                          "90starsbrk", "codeoflifebrk", "saab", "zurklezsolidbrk", "pothana2000",
                                          "kacsttitle", "fauxsnowbrk", "lohitgujarati", "tetricidebrk", "lohittamil",
                                          "kacsttitlel", "droidsansthai", "dblayer4brk", "bitblocksttfbrk",
                                          "pindownbrk", "droidarabicnaskh"]:  # font is no good
                    continue
                # if text == "":
                #    text=font.split("/")[-1]
                img2.dl().selectFont(font)
                y = random.randint(0, h)
                x = random.randint(0, w)
                fs = int(w / 1.1)
                fontSize = random.randint(random.randint(fs, w), int(w * 1.1))
                img2.dl().setFontSize(fontSize)  # font size between 2 and 100
                (w1, h1) = img2.dl().textDimensions(text)
                k = 0
                skip_this_one = False
                while x + w1 > w * 1.1 or y + h1 > h * 1.1:  # 1.2 for allowing text to be partly outside the box
                    k += 1
                    y = random.randint(0, h)
                    x = random.randint(0, w)
                    # if k>5:
                    #    y=random.randint(0,h//3+int(random.random()*0.6*h))
                    #    x=random.randint(0,int(w*random.random()*0.4))
                    if k > 500:
                        skip_this_one = True  # the font is too big for the image dimensions
                        print("skipping the big font:" + str(font))
                        break
                    fontSize = int(0.95 * fontSize)
                    if fontSize < w // 5:
                        fontSize = random.randint(random.randint(fs, w - 1), w)
                    fs -= 2
                    if fs < 3:
                        fs = random.randint(2, w - 1)
                    img2.dl().setFontSize(fontSize)
                    (w1, h1) = img2.dl().textDimensions(text)
                if skip_this_one:
                    img2.clearLayers()
                    newImg.clearLayers()
                    continue
                img2.drawText(text, x, y, colors[i], fontSize)
                if random.randint(0, 10) < 2:
                    img = newImg + img2.applyLayers()
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                if random.randint(0, 10) < 1:
                    img = newImg + img2.applyLayers().warp(clockwise_quadrilateral(w, h))
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                if random.randint(0, 10) < 1:
                    img = newImg + img2.applyLayers().rotate(random.randint(-90, 90))
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                if random.randint(0, 10) < 8:  # warp
                    img = newImg + img2.applyLayers().rotate(random.randint(-90, 90)).warp(
                            clockwise_quadrilateral(w, h))
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                if random.randint(0, 10) < 5:  # salt&pepper
                    img = add_noise(img, random.random() * 0.0025)
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                # img=img.smooth(sigma=random.random()*3,spatial_sigma=random.random()*3)
                # if random.randint(0,10)<1:
                #    img=add_noise(img,random.random()*0.001)
                img2.clearLayers()

                if random.randint(0, 10) < 11:  # always true
                    img = img.blur((random.randint(1, random.randint(2, fs // 5 + 4)),
                                    random.randint(1, random.randint(2, fs // 5 + 4))))
                    if random.randint(0, 10) < 7:
                        img = img.invert()
                    train_csv.write(text + "," + ",".join(str(x) for x in misc.img_to_1d_gray(img)) + "\n")
                    # img.show()
                    # time.sleep(0.5)
                    # file_to_save = fname[:-4:] + "_" + text + str(font.replace("/", "_")) + fname[-4::]
                    # print(file_to_save)
                    # img.save(path + file_to_save)  # str(font.replace("/","_"))+fname[-4::])

                    # Popen(['/home/agp/Desktop/scenetext/code/DetectText/./DetectText',
                    #       file_to_save,
                    #       '/home/agp/Desktop/scenetext/code/DetectText/result.png', '1'])
                    # Popen('rm temp.png',shell=True,stdin=PIPE,stdout=PIPE)
                    # print("swt:")
                    # Image('SWT.png').show()
                    # time.sleep(5)

                # else:
                #    img.save(sys.argv[1][:-4:]+"_"+str(font)+sys.argv[1][-4::])
                # newImg.clearLayers()
                i += 1
                # if text==font.split("/")[-1]:
                #    text=""
                # images=[i for i in Popen('find `pwd` -type f',shell=True,stdin=PIPE,stdout=PIPE).stdout.read().split('\n') if i[-9:-4]=='00001']

# generate_letter_images()
