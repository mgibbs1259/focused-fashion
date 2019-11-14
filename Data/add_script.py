import os
import json

from Data.load_data import download_image


ids_to_add_back = [4025, 4761, 5209, 5397, 6980, 7421, 10370, 15035, 16593, 17043, 17325, 17723, 18290, 18371, 18499,
                   22034, 22657, 23691, 24917, 24945, 25684, 26466, 26751, 27842, 27922, 32016, 32320, 33128, 33188,
                   35989, 36422, 36903, 39054, 41664, 41956, 42917, 43349, 43685, 44421, 46975, 47114, 48111, 48864,
                   49578, 49681, 51389, 51536, 51767, 53491, 53700, 54652, 56290, 57813, 59003, 60347, 60617, 61188,
                   63101, 63875, 68602, 70814, 71810, 73339, 74033, 74998, 75505, 76233, 76809, 80089, 82442, 85677,
                   86080, 86150, 88548, 90700, 93986, 95747, 101518, 103637, 104953, 105899, 111037, 113760, 120726,
                   121375, 122883, 124000, 124602, 130958, 136656, 136723, 136755, 137984, 138133, 138275, 139440,
                   139630, 140366, 144691, 147359, 149251, 150880, 164652, 166316, 166646, 170559, 170863, 171555,
                   172679, 173894, 181038, 185958, 188010, 194147, 194806, 195450, 196673, 199922, 201519, 203886,
                   209791, 213136, 215975, 221386, 225842, 227956, 230161, 231454, 231906, 232377, 238809, 238852,
                   242294, 244090, 245266, 247580, 253285, 256762, 257010, 259156, 262625, 263565, 263855, 263890,
                   264750, 275170, 281865, 283038, 283552, 289043, 290078, 293349, 295735, 295855, 296954, 298127,
                   307046, 312840, 313133, 314272, 317393, 319303, 323044, 323942, 327624, 330160, 334708, 334849,
                   335446, 341381, 341620, 341697, 343349, 344346, 352029, 364155, 364874, 369844, 371094, 378664,
                   383209, 384895, 385742, 385964, 387825, 389296, 394076, 394423, 396905, 399349, 401007, 403210,
                   405840, 414053, 414786, 414960, 415941, 417574, 419155, 420592, 420908, 421524, 422174, 423191,
                   423417, 424093, 424596, 426121, 426725, 427986, 430809, 431211, 431413, 435002, 435742, 438570,
                   443971, 444251, 448629, 449134, 450240, 450491, 452684, 459737, 460940, 462197, 462591, 465350,
                   467673, 467699, 467873, 468055, 468095, 468296, 468414, 468585, 469001, 469015, 469627, 469732,
                   469926, 469967, 470105, 470136, 470180, 470373, 470418, 470499, 470805, 470973, 471253, 471617,
                   471690, 471833, 471859, 471910, 472029, 472416, 472489, 472569, 472846, 472881, 472895, 473115,
                   473251, 473288, 473357, 473472, 473528, 473811, 473933, 474006, 474194, 474197, 474225, 474629,
                   474680, 474840, 474855, 474959, 475143, 475182, 475928, 475969, 476024, 476419, 476456, 476648,
                   476839, 476911, 476919, 477140, 477264, 477386, 477515, 477571, 477763, 477894, 478066, 478228,
                   478612, 478680, 478728, 478778, 478894, 479173, 479318, 479365, 479387, 479535, 479607, 479644,
                   479733, 479791, 479800, 479844, 479845, 480296, 480401, 480493, 480816, 481343, 481429, 481430,
                   481478, 481635, 481659, 481775, 481943, 481954, 482112, 482158, 482204, 482495, 482507, 482543,
                   482652, 482690, 482751, 482916, 482993, 483022, 483092, 483261, 483375, 483596, 483667, 484350,
                   484369, 484820, 485141, 485235, 485314, 485505, 485573, 485580, 485678, 485749, 485751, 485904,
                   485997, 486057, 486182, 486316, 486777, 487088, 487166, 487402, 487430, 487468, 487503, 487645,
                   487811, 487974, 488440, 488447, 489031, 489150, 489182, 489295, 489335, 489340, 489346, 489755,
                   490128, 490195, 490374, 490401, 490682, 490764, 490845, 490862, 491012, 491236, 491256, 491735,
                   492103, 492387, 492440, 492647, 493002, 493056, 493065, 493151, 493337, 493372, 493507, 493571,
                   494200, 494297, 494431, 494981, 495121, 495215, 495299, 495363, 495449, 495739, 496111, 496406,
                   496553, 496692, 496777, 496795, 497251, 497254, 497360, 497558, 497573, 497753, 497828, 497943,
                   497998, 498278, 498633, 498982, 499108, 499168, 499260, 499596]


data_dir = "/home/ubuntu/Final-Project-Group8/Data/" # Look at this and ensure it's path to train.json
_outdir = "/home/ubuntu/Final-Project-Group8/Data/output_train" # Change this to train img dir


with open(os.path.join(data_dir, 'train.json')) as f:
    data = json.load(f)

fname_urls = []
for image in data['images']:
    url = image['url']
    f = image['imageId']
    for element in ids_to_add_back:
        if str(element) == f:
            fname = os.path.join(_outdir, "{}.jpg".format(f))
            fname_urls.append((fname, url))

for f in fname_urls:
    print("Downloading")
    download_image(f)

print(len(os.listdir(_outdir)))