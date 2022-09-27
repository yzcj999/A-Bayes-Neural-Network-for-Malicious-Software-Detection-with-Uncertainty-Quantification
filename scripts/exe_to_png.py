from PIL import Image
import os
def getBinaryData(filename):
    """
    Extract byte values from binary executable file and store them into list
    :param filename: executable file name
    :return: byte value list
    """
    binary_values = []

    with open(filename, 'rb') as fileobject:

        # read file byte by byte
        data = fileobject.read(1)

        while data != b'':
            binary_values.append(ord(data))
            data = fileobject.read(1)
    return binary_values

def createGreyScaleImage(filename, width=None):
    greyscale_data  = getBinaryData(filename)
    size            = get_size(len(greyscale_data), width)
    # try:
    image = Image.new('L',size)
    image.putdata(greyscale_data)
      # setup output filename
    dirname     = 'img_out'
    name, _     = os.path.splitext(filename)
    name        = os.path.basename(name)
    imagename   = dirname + os.sep + name + '_'+ '.png'
    os.makedirs(os.path.dirname(imagename), exist_ok=True)
    img_path = imagename
    image.save(imagename)
    print('The file', imagename, 'saved.')
    return img_path
    # except Exception as err:
    # 	  print(err)
  


def get_size(data_length, width=None):
    # source Malware images: visualization and automatic classification by L. Nataraj
    # url : http://dl.acm.org/citation.cfm?id=2016908

    if width is None: # with don't specified any with value

        size = data_length

        if (size < 10240):
            width = 32
        elif (10240 <= size <= 10240 * 3):
            width = 64
        elif (10240 * 3 <= size <= 10240 * 6):
            width = 128
        elif (10240 * 6 <= size <= 10240 * 10):
            width = 256
        elif (10240 * 10 <= size <= 10240 * 20):
            width = 384
        elif (10240 * 20 <= size <= 10240 * 50):
            width = 512
        elif (10240 * 50 <= size <= 10240 * 100):
            width = 768
        else:
            width = 1024

        height = int(size / width) + 1

    else:
        width  = int(math.sqrt(data_length)) + 1
        height = width

    return (width, height)