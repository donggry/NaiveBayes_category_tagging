import numpy as np
import random
import sys, math
from datetime import datetime
import pickle
from operator import itemgetter
import nltk
import string
from nltk.stem.snowball import SnowballStemmer

class Classifier:
    def __init__(self, featureGenerator):
        self.featureGenerator = featureGenerator
        self._C_SIZE = 0
        self._V_SIZE = 0
        self._classes_list = []
        self._classes_dict = {}
        self._vocab = {}
        #self._vocab_doc = {} # add by Hana
        #self._vocab_after = {}  # add by Hana

        # add by Eunsoo
        #self.re_class_dic = {}
        self.myweight = []
        self.myweight_dic = {}
        self.re_vocab = {}

    def setClasses(self, trainingData, level):
        for (label, line) in trainingData:
            if label not in self._classes_dict.keys():
                #self._classes_dict[label] = len(self._classes_list)
                self._classes_list.append(label)

        #cat_dict_list = open('cat_dict.txt', 'r').read().split('\n')
        dict1 = {'Grocery, Household & Pets': 7, "Women's Fashion": 1, 'For the Home': 9, 'Entertainment': 2, 'Health & Beauty': 6, "Men's Fashion": 8, 'Baby, Kids & Toys': 10, 'Jewelry & Watches': 3, 'Electronics': 0, 'Auto & Home Improvement': 5, 'Sports & Outdoors': 4}
        dict2 = {'Auto & Home Improvement / Patio & Garden': 75, 'Baby, Kids & Toys / Girls Fashion': 30, 'Baby, Kids & Toys / Bedding & Bath': 48, 'Health & Beauty / Personal Care': 29, 'Electronics / Car Electronics & GPS': 76, 'For the Home / Home Decor': 44, 'Electronics / Cell Phones & Accessories': 35, 'Electronics / Musical Instruments': 15, 'Sports & Outdoors / Outdoors': 36, "Women's Fashion / Plus Size Clothing": 49, 'Electronics / Video Games': 52, 'Grocery, Household & Pets / Candy & Sweets': 43, 'Auto & Home Improvement / Automotive': 12, 'For the Home / Furniture': 6, 'For the Home / Patio & Garden': 37, 'Electronics / Portable Audio': 16, 'Health & Beauty / Hair Care': 7, 'Health & Beauty / Cosmetics': 0, 'Sports & Outdoors / Exercise & Fitness': 53, 'Health & Beauty / Sexual Wellness': 1, "Women's Fashion / Intimates": 13, 'Grocery, Household & Pets / Household Essentials': 24, 'Sports & Outdoors / Clothing & Shoes': 61, 'Health & Beauty / Skin Care': 25, 'Health & Beauty / Massage & Relaxation': 58, 'Entertainment / Video Games': 73, 'For the Home / Luggage': 60, 'For the Home / Bath': 47, 'For the Home / Heating & Cooling': 63, "Women's Fashion / Shoes": 19, 'Grocery, Household & Pets / Tobacco': 55, 'Jewelry & Watches / Fine Metal Jewelry': 56, 'Electronics / Computers & Tablets': 14, 'Entertainment / Magazines': 74, 'Jewelry & Watches / Jewelry Accessories & Storage': 71, 'Grocery, Household & Pets / Alcohol': 67, "Men's Fashion / Clothing": 28, 'Sports & Outdoors / Recreation': 64, "Jewelry & Watches / Men's Jewelry": 62, 'Grocery, Household & Pets / Food': 50, 'Sports & Outdoors / Team Sports': 65, 'Electronics / Office & School Supplies': 2, 'Health & Beauty / Bath & Body': 57, 'Health & Beauty / Fragrance': 27, "Men's Fashion / Shoes": 17, 'Jewelry & Watches / Gemstone & Pearl Jewelry': 69, 'Sports & Outdoors / Golf': 21, 'Jewelry & Watches / Watches': 39, 'Electronics / Camera, Video & Surveillance': 33, "Women's Fashion / Maternity Clothing": 78, 'Baby, Kids & Toys / Maternity': 68, 'Auto & Home Improvement / Home Appliances': 77, 'Baby, Kids & Toys / Boys Fashion': 34, 'For the Home / Home Appliances': 42, 'Baby, Kids & Toys / Baby Care': 46, 'Auto & Home Improvement / Home Improvement': 20, 'For the Home / Art': 10, 'Health & Beauty / Vitamins & Supplements': 26, 'Sports & Outdoors / Fan Shop': 38, 'Jewelry & Watches / Diamond Jewelry': 23, 'Entertainment / Music': 54, 'Electronics / Software': 41, 'Sports & Outdoors / Cycling': 70, 'Entertainment / Books': 72, 'Baby, Kids & Toys / Health & Safety': 59, 'Grocery, Household & Pets / Pets': 9, "Women's Fashion / Accessories": 32, "Women's Fashion / Clothing": 4, "Men's Fashion / Accessories": 31, 'For the Home / Mattresses & Accessories': 66, 'Health & Beauty / Health Care': 11, 'For the Home / Bedding': 40, 'For the Home / Storage & Organization': 45, 'Jewelry & Watches / Fashion Jewelry': 22, 'Grocery, Household & Pets / Beverages': 51, 'Baby, Kids & Toys / Toys': 3, 'Entertainment / Movies & TV': 5, 'For the Home / Kitchen & Dining': 8, 'Electronics / Television & Home Theater': 18}
        dict3 = {'Health & Beauty / Cosmetics / Face Makeup\r': 250, "Men's Fashion / Shoes / Sandals\r": 225, 'Jewelry & Watches / Fashion Jewelry / Collections & Sets\r': 318, 'For the Home / Bedding / Duvet Covers\r': 21, "Women's Fashion / Clothing / Pants\r": 180, 'For the Home / Home Decor / Home Accents\r': 319, 'Jewelry & Watches / Diamond Jewelry / Necklaces\r': 45, 'Electronics / Office & School Supplies / Printer Ink & Toner\r': 170, "Women's Fashion / Clothing / Costumes\r": 93, 'Grocery, Household & Pets / Food / Snack Foods\r': 110, 'Sports & Outdoors / Team Sports / Soccer\r': 293, 'Sports & Outdoors / Team Sports / Baseball & Softball\r': 424, 'Health & Beauty / Sexual Wellness / Bondage & Fetish\r': 321, 'For the Home / Home Appliances / Small Appliances\r': 129, 'For the Home / Art / Framed Art\r': 49, 'Sports & Outdoors / Cycling / Clothing & Footwear\r': 448, 'Electronics / Portable Audio / In-Ear & Earbud Headphones\r': 39, 'Auto & Home Improvement / Home Appliances / Dishwashers\r': 462, 'Auto & Home Improvement / Home Improvement / Tool Storage\r': 151, "Men's Fashion / Shoes / Oxfords\r": 161, 'Electronics / Cell Phones & Accessories / Screen Protectors\r': 247, 'Electronics / Computers & Tablets / Computer Accessories\r': 25, 'Electronics / Musical Instruments / Drums & Percussion\r': 248, 'Electronics / Computers & Tablets / Laptops\r': 181, 'Auto & Home Improvement / Home Improvement / Heating & Cooling\r': 341, 'Health & Beauty / Health Care / Humidifiers\r': 203, 'Electronics / Portable Audio / On-Ear & Over-Ear Headphones\r': 280, 'Health & Beauty / Health Care / Medical Braces\r': 312, 'Jewelry & Watches / Fine Metal Jewelry / Bracelets & Bangles\r': 162, "Men's Fashion / Clothing / Pants\r": 205, 'Jewelry & Watches / Diamond Jewelry / Novelty\r': 491, 'Electronics / Cell Phones & Accessories / Cables, Chargers & Adapters\r': 81, 'For the Home / Home Decor / Candles & Holders\r': 131, 'Sports & Outdoors / Team Sports / Lacrosse\r': 259, 'For the Home / Art / Photography\r': 185, 'Electronics / Musical Instruments / Guitars\r': 202, 'Entertainment / Movies & TV / Television\r': 12, 'For the Home / Patio & Garden / Bird Feeders & Food\r': 371, 'Baby, Kids & Toys / Toys / Pretend Play\r': 286, 'Jewelry & Watches / Diamond Jewelry / Rings\r': 349, 'Electronics / Software / Business & Home Office\r': 417, 'For the Home / Bedding / Down & Alternative Comforters\r': 343, 'Grocery, Household & Pets / Candy & Sweets / Fruit & Nut\r': 495, 'Auto & Home Improvement / Home Improvement / Power Tools\r': 164, "Women's Fashion / Clothing / Activewear\r": 143, 'For the Home / Home Appliances / Vacuums & Floor Care\r': 67, 'Jewelry & Watches / Fashion Jewelry / Necklaces\r': 379, 'Health & Beauty / Massage & Relaxation / Handheld Massagers\r': 118, "Men's Fashion / Clothing / Shorts\r": 65, 'Baby, Kids & Toys / Toys / Outdoor Play\r': 155, 'Electronics / Portable Audio / Bluetooth & Wireless Speakers\r': 191, "Men's Fashion / Clothing / Socks\r": 272, 'Health & Beauty / Personal Care / Oral Care\r': 135, 'Jewelry & Watches / Fashion Jewelry / Novelty\r': 335, 'Health & Beauty / Sexual Wellness / Adult Toys for Couples\r': 387, 'Baby, Kids & Toys / Toys / Dolls & Action Figures\r': 169, "Men's Fashion / Accessories / Wallets & Money Clips\r": 57, 'Entertainment / Music / Rap\r': 474, 'Entertainment / Movies & TV / History & Documentary\r': 192, 'Health & Beauty / Skin Care / Hair Removal\r': 395, 'For the Home / Home Appliances / Irons & Garment Care\r': 43, 'Health & Beauty / Personal Care / Shaving & Grooming\r': 11, 'Electronics / Musical Instruments / Accessories\r': 153, "Men's Fashion / Clothing / Activewear\r": 176, 'For the Home / Heating & Cooling / Dehumidifiers\r': 370, "Women's Fashion / Clothing / Outerwear & Suiting\r": 46, 'Electronics / Television & Home Theater / Blu Ray & DVD Players\r': 231, "Baby, Kids & Toys / Girls Fashion / Girls' Shoes\r": 116, 'Jewelry & Watches / Fine Metal Jewelry / Necklaces\r': 306, "Men's Fashion / Shoes / Shoe Accessories\r": 443, "Jewelry & Watches / Watches / Men's Watches\r": 296, 'For the Home / Bath / Bath Accessories & Sets\r': 327, 'Electronics / Car Electronics & GPS / Car Security & Remote Start\r': 460, "Jewelry & Watches / Men's Jewelry / Novelty\r": 394, 'Electronics / Software / Programming\r': 457, 'For the Home / Patio & Garden / Saunas Spas & Hot Tubs\r': 471, "Men's Fashion / Accessories / Bags\r": 37, 'Health & Beauty / Vitamins & Supplements / Weight Loss\r': 20, 'Auto & Home Improvement / Home Improvement / Garage\r': 345, 'Electronics / Camera, Video & Surveillance / Point & Shoots\r': 273, 'Entertainment / Video Games / Game Gear & Novelties\r': 455, "Women's Fashion / Shoes / Evening\r": 103, 'For the Home / Luggage / Hardside\r': 388, 'Baby, Kids & Toys / Maternity / Tops\r': 6, 'Health & Beauty / Health Care / Healthy Living\r': 53, 'For the Home / Patio & Garden / Gardening & Lawn Care\r': 95, 'Grocery, Household & Pets / Candy & Sweets / Bakery\r': 206, 'Entertainment / Magazines / Business & Finance\r': 507, "Women's Fashion / Clothing / Sweaters & Cardigans\r": 87, 'Baby, Kids & Toys / Boys Fashion / Watches\r': 204, 'For the Home / Furniture / Accent Furniture\r': 376, 'Health & Beauty / Vitamins & Supplements / Vitamins\r': 283, 'For the Home / Home Decor / Picture Frames\r': 322, "Women's Fashion / Intimates / Socks & Hosiery\r": 19, 'Health & Beauty / Cosmetics / Lips\r': 142, "Women's Fashion / Plus Size Clothing / Intimates\r": 324, 'Electronics / Television & Home Theater / TVs\r': 190, 'Auto & Home Improvement / Automotive / Exterior Accessories\r': 62, 'Health & Beauty / Massage & Relaxation / Head Massagers\r': 418, "Women's Fashion / Accessories / Gloves\r": 257, 'Entertainment / Books / Hobbies\r': 337, 'Jewelry & Watches / Fashion Jewelry / Bracelets & Bangles\r': 223, 'Auto & Home Improvement / Automotive / Car Care\r': 302, 'Health & Beauty / Skin Care / Sun Care & Tanning\r': 188, 'For the Home / Home Appliances / Wine Coolers\r': 420, 'Sports & Outdoors / Exercise & Fitness / Cardio Training\r': 163, 'For the Home / Storage & Organization / Bathroom\r': 369, "For the Home / Furniture / Baby & Kid's Furniture\r": 98, 'Grocery, Household & Pets / Tobacco / Vaporizers & E-Cigs\r': 256, 'Entertainment / Movies & TV / Sci-fi & Fantasy\r': 14, "Women's Fashion / Maternity Clothing / Intimates\r": 444, 'Sports & Outdoors / Exercise & Fitness / Books & Magazines\r': 427, "Women's Fashion / Clothing / Skirts\r": 216, 'For the Home / Furniture / Kitchen & Dining Furniture\r': 100, "Women's Fashion / Accessories / Sunglasses & Eyewear\r": 85, 'Health & Beauty / Vitamins & Supplements / Multi & Prenatal Vitamins\r': 127, 'For the Home / Bedding / Blankets & Throws\r': 173, 'Grocery, Household & Pets / Beverages / Sports Drinks\r': 446, 'Electronics / Office & School Supplies / Paper & Stationery\r': 254, "Men's Fashion / Shoes / Dress Shoes\r": 182, "Women's Fashion / Shoes / Loafers & Slip-Ons\r": 264, 'Jewelry & Watches / Fine Metal Jewelry / Novelty\r': 367, 'Electronics / Office & School Supplies / School Supplies\r': 76, 'Grocery, Household & Pets / Household Essentials / Food Storage\r': 22, 'Sports & Outdoors / Exercise & Fitness / Balance & Recovery\r': 435, 'Grocery, Household & Pets / Beverages / Water\r': 473, 'For the Home / Kitchen & Dining / Kitchen Appliances\r': 493, 'Health & Beauty / Bath & Body / Accessories\r': 0, 'Jewelry & Watches / Diamond Jewelry / Wedding & Engagement\r': 276, 'For the Home / Home Decor / Rugs\r': 132, 'Health & Beauty / Skin Care / Treatments & Serums\r': 42, "Baby, Kids & Toys / Girls Fashion / Girls' Clothing\r": 121, 'Health & Beauty / Sexual Wellness / Lubricants & Sexual Enhancers\r': 196, 'Auto & Home Improvement / Home Improvement / Bathroom Faucets\r': 506, 'Health & Beauty / Health Care / Health Monitors\r': 112, 'For the Home / Patio & Garden / Outdoor Power Equipment\r': 220, 'For the Home / Luggage / Carry-Ons\r': 334, 'Baby, Kids & Toys / Toys / Arts & Crafts\r': 146, "Men's Fashion / Clothing / Outerwear & Jackets\r": 56, 'Jewelry & Watches / Watches / Watch Accessories\r': 168, 'For the Home / Art / Prints & Decals\r': 140, "Women's Fashion / Clothing / Shorts & Capris\r": 90, 'Health & Beauty / Health Care / Pain Relief\r': 208, 'Health & Beauty / Sexual Wellness / Anal Toys\r': 187, 'Health & Beauty / Health Care / First Aid\r': 380, "Men's Fashion / Clothing / Lounge & Sleepwear\r": 89, 'Entertainment / Music / Rock\r': 396, "Baby, Kids & Toys / Boys Fashion / Boys' Accessories\r": 243, 'For the Home / Bedding / Sheets\r': 34, 'Electronics / Television & Home Theater / Home Theater Accessories\r': 8, 'Electronics / Computers & Tablets / Desktops, Monitors, & All-In-Ones\r': 244, 'For the Home / Storage & Organization / Laundry\r': 166, 'Health & Beauty / Massage & Relaxation / Acupuncture & Acupressure\r': 404, "Jewelry & Watches / Men's Jewelry / Earrings\r": 269, "For the Home / Luggage / Kid's Travel Bags\r": 177, 'For the Home / Art / Canvas\r': 16, 'Health & Beauty / Bath & Body / Hands & Feet\r': 391, 'Grocery, Household & Pets / Household Essentials / Home Fragrance & Air Care\r': 441, "Men's Fashion / Accessories / Pocket Squares\r": 482, 'For the Home / Patio & Garden / Grills & Outdoor Cooking\r': 292, 'Grocery, Household & Pets / Beverages / Coffee\r': 97, 'Electronics / Cell Phones & Accessories / Bluetooth Devices\r': 227, 'For the Home / Mattresses & Accessories / Memory Foam Mattresses\r': 71, 'Health & Beauty / Personal Care / Incontinence\r': 467, 'Grocery, Household & Pets / Candy & Sweets / Chocolate\r': 464, 'For the Home / Furniture / Living Room Furniture\r': 28, 'For the Home / Storage & Organization / Office\r': 326, 'Grocery, Household & Pets / Alcohol / Beer\r': 500, "Jewelry & Watches / Men's Jewelry / Collections & Sets\r": 430, 'Sports & Outdoors / Recreation / Lawn Games\r': 378, "Sports & Outdoors / Clothing & Shoes / Men's Activewear\r": 213, 'Baby, Kids & Toys / Health & Safety / Baby Monitors\r': 468, "Entertainment / Music / Kid's Music\r": 504, 'Health & Beauty / Massage & Relaxation / Total Body Massagers\r': 241, "Health & Beauty / Fragrance / Men's Fragrance\r": 66, 'Health & Beauty / Hair Care / Hair Accessories\r': 157, 'Electronics / Portable Audio / Docks, Radios & Boom Boxes\r': 83, 'Sports & Outdoors / Outdoors / Action Sports\r': 77, "Women's Fashion / Clothing / Swimwear\r": 195, "Baby, Kids & Toys / Girls Fashion / Girls' Accessories\r": 346, 'Grocery, Household & Pets / Candy & Sweets / Gum & Mints\r': 156, 'For the Home / Kitchen & Dining / Serveware\r': 212, 'Grocery, Household & Pets / Alcohol / Mixers & Ready To Drink\r': 488, 'Electronics / Office & School Supplies / Home Office Furniture\r': 199, 'Baby, Kids & Toys / Toys / Electronic Toys\r': 265, "Baby, Kids & Toys / Boys Fashion / Boys' Clothing\r": 263, 'Health & Beauty / Personal Care / Foot Care\r': 147, 'Auto & Home Improvement / Automotive / Interior Accessories\r': 84, 'For the Home / Storage & Organization / Closet\r': 41, 'Electronics / Office & School Supplies / Shredders\r': 386, "Women's Fashion / Intimates / Lounge & Sleepwear\r": 139, 'Auto & Home Improvement / Home Improvement / Home Safety & Security\r': 17, 'Auto & Home Improvement / Home Improvement / Hardware\r': 277, 'For the Home / Home Appliances / Dishwashers\r': 470, "Men's Fashion / Accessories / Scarves, Hats & Gloves\r": 79, 'Baby, Kids & Toys / Toys / Building Sets & Blocks\r': 55, 'Health & Beauty / Massage & Relaxation / Massage Oils, Aromatherapy & Lotions\r': 44, 'Grocery, Household & Pets / Beverages / Powdered Drink Mixes\r': 416, 'For the Home / Kitchen & Dining / Dinnerware\r': 136, 'For the Home / Luggage / Duffel Bags\r': 260, "Women's Fashion / Accessories / Wallets\r": 207, 'For the Home / Kitchen & Dining / Flatware\r': 431, 'Sports & Outdoors / Golf / Golf Balls\r': 342, 'Jewelry & Watches / Fine Metal Jewelry / Earrings\r': 422, "Women's Fashion / Plus Size Clothing / Bottoms\r": 193, 'Entertainment / Music / Soundtrack\r': 479, 'Health & Beauty / Bath & Body / Body Cleansers\r': 109, 'Electronics / Office & School Supplies / Packing & Mailing\r': 237, 'Sports & Outdoors / Golf / Golf Bags and Cart\r': 405, 'Health & Beauty / Skin Care / Cleanse\r': 40, 'Electronics / Television & Home Theater / Projectors & Screens\r': 133, 'For the Home / Bath / Bath Rugs\r': 251, 'Health & Beauty / Cosmetics / Nails\r': 128, 'Health & Beauty / Cosmetics / Makeup Palettes & Sets\r': 27, 'Sports & Outdoors / Fan Shop / MLS\r': 289, "Women's Fashion / Clothing / Dresses\r": 106, 'Electronics / Office & School Supplies / Scanners\r': 408, 'Health & Beauty / Massage & Relaxation / Massage Accessories\r': 88, 'Baby, Kids & Toys / Bedding & Bath / Crib Sets\r': 381, 'Baby, Kids & Toys / Bedding & Bath / Potty Training\r': 496, "Men's Fashion / Shoes / Slippers\r": 201, 'Health & Beauty / Cosmetics / Bags & Cases\r': 393, 'Health & Beauty / Health Care / Daily Living Aids\r': 236, 'Entertainment / Movies & TV / Drama\r': 2, 'Auto & Home Improvement / Home Appliances / Vacuums & Floor Care\r': 442, 'Electronics / Computers & Tablets / Tablets & E-Readers\r': 78, 'Auto & Home Improvement / Home Improvement / Paint & Wallpapers\r': 194, "Women's Fashion / Clothing / Leggings\r": 7, 'Entertainment / Music / Country\r': 501, 'For the Home / Bath / Beach Towels\r': 359, 'For the Home / Bedding / Comforters\r': 86, "Jewelry & Watches / Watches / Women's Watches\r": 451, 'Sports & Outdoors / Team Sports / Football\r': 415, 'Health & Beauty / Personal Care / Body Treatments\r': 99, 'Baby, Kids & Toys / Health & Safety / Safety\r': 74, 'Jewelry & Watches / Fashion Jewelry / Earrings\r': 275, "Women's Fashion / Shoes / Athletic\r": 300, 'Electronics / Office & School Supplies / Organization\r': 144, 'Health & Beauty / Bath & Body / Bath Soaks & Bubble Baths\r': 38, 'Baby, Kids & Toys / Maternity / Bottoms\r': 311, 'For the Home / Home Appliances / Washers & Dryers\r': 425, "Men's Fashion / Clothing / Costumes\r": 266, 'For the Home / Luggage / Luggage Sets\r': 226, 'Grocery, Household & Pets / Beverages / Coconut Water\r': 377, "Women's Fashion / Shoes / Fashion Sneakers\r": 148, 'Entertainment / Music / Pop\r': 445, 'Electronics / Cell Phones & Accessories / Cases\r': 61, 'Grocery, Household & Pets / Beverages / Hot Cocoa\r': 498, 'For the Home / Kitchen & Dining / Kitchen Towels & Aprons\r': 429, 'For the Home / Kitchen & Dining / Drinkware\r': 29, 'Grocery, Household & Pets / Beverages / Tea\r': 107, 'Entertainment / Magazines / Hobbies & Crafts\r': 499, 'Entertainment / Video Games / Game Consoles\r': 268, 'Grocery, Household & Pets / Pets / Cats\r': 138, "Women's Fashion / Plus Size Clothing / Outerwear & Suiting\r": 113, 'For the Home / Bedding / Baby Bedding\r': 271, 'Baby, Kids & Toys / Maternity / Nursing\r': 484, 'Electronics / Cell Phones & Accessories / Cell Phone Accessories\r': 82, 'For the Home / Furniture / Bedroom Furniture\r': 10, 'Grocery, Household & Pets / Food / Breakfast Foods\r': 294, 'For the Home / Bath / Bathroom Scales\r': 362, 'For the Home / Bath / Shower Curtains & Liners\r': 303, 'Sports & Outdoors / Team Sports / Volleyball\r': 494, 'Health & Beauty / Health Care / Sleep Aids\r': 336, 'Jewelry & Watches / Fashion Jewelry / Rings\r': 352, 'Baby, Kids & Toys / Bedding & Bath / Bath Tubs & Seats\r': 281, 'Health & Beauty / Hair Care / Shampoo & Conditioner\r': 23, 'Baby, Kids & Toys / Baby Care / Baby Gear\r': 325, "Women's Fashion / Shoes / Boots\r": 1, 'Electronics / Camera, Video & Surveillance / Security & Surveillance\r': 52, "Women's Fashion / Shoes / Sandals\r": 59, 'Entertainment / Books / Non-Fiction\r': 436, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Bracelets & Bangles\r': 329, 'For the Home / Furniture / Bathroom Furniture\r': 505, 'For the Home / Home Appliances / Refrigerators\r': 483, "Women's Fashion / Shoes / Slippers\r": 200, 'Jewelry & Watches / Jewelry Accessories & Storage / Boxes & Holders\r': 339, 'Electronics / Video Games / Video Game Accessories\r': 184, 'Health & Beauty / Bath & Body / Body Moisturizers\r': 297, "Men's Fashion / Clothing / Sweaters & Sweatshirts\r": 48, 'Baby, Kids & Toys / Maternity / Dresses\r': 340, 'For the Home / Bedding / Mattress Toppers & Pads\r': 361, 'Electronics / Camera, Video & Surveillance / Action Cameras & Drones\r': 171, 'Grocery, Household & Pets / Beverages / Soft Drinks\r': 476, 'Electronics / Video Games / Games (92)\r': 299, "Women's Fashion / Intimates / Lingerie\r": 105, 'Sports & Outdoors / Fan Shop / Memorabilia\r': 478, 'Sports & Outdoors / Fan Shop / NASCAR\r': 487, "Men's Fashion / Accessories / Belts & Suspenders\r": 287, 'Health & Beauty / Hair Care / Hair Color\r': 158, 'Auto & Home Improvement / Home Improvement / Lighting\r': 150, 'Electronics / Cell Phones & Accessories / Mounts & Stands\r': 419, 'Sports & Outdoors / Outdoors / Hunting\r': 351, 'Grocery, Household & Pets / Candy & Sweets / Assortments\r': 439, 'Entertainment / Video Games / Game Consoles (53)\r': 458, 'Auto & Home Improvement / Home Improvement / Hand Tools\r': 104, "Women's Fashion / Shoes / Flats\r": 50, 'Grocery, Household & Pets / Pets / Dogs\r': 101, 'Grocery, Household & Pets / Pets / Fish\r': 447, 'Grocery, Household & Pets / Tobacco / Cigars\r': 486, 'Jewelry & Watches / Fine Metal Jewelry / Rings\r': 331, 'For the Home / Luggage / Suitcases\r': 215, 'Auto & Home Improvement / Automotive / Car Electronics\r': 94, 'For the Home / Furniture / Home Office Furniture\r': 30, 'Electronics / Television & Home Theater / Streaming Media Players & Antennas\r': 368, 'Health & Beauty / Massage & Relaxation / Pulse Massagers\r': 209, 'Grocery, Household & Pets / Pets / Birds\r': 409, 'For the Home / Bath / Bath Storage & Caddies\r': 310, 'Grocery, Household & Pets / Pets / Small Animals\r': 358, 'Sports & Outdoors / Exercise & Fitness / Fitness Technology\r': 313, 'Auto & Home Improvement / Home Improvement / Flooring\r': 330, 'Entertainment / Books / Fiction & Literature\r': 390, 'Electronics / Computers & Tablets / Networking & Wireless\r': 401, "Health & Beauty / Fragrance / Women's Fragrance\r": 70, 'Health & Beauty / Personal Care / Pregnancy & Fertility\r': 178, 'Sports & Outdoors / Outdoors / Winter Sports\r': 384, 'Health & Beauty / Vitamins & Supplements / Sports Nutrition\r': 31, 'For the Home / Bedding / Quilts & Bedspreads\r': 198, "Jewelry & Watches / Men's Jewelry / Bracelets\r": 320, 'Auto & Home Improvement / Home Improvement / Shower Heads\r': 315, 'Electronics / Portable Audio / Headphone Accessories\r': 453, 'Health & Beauty / Vitamins & Supplements / Herbal Remedies & Teas\r': 102, "Men's Fashion / Clothing / Button-Down Shirts\r": 58, 'Baby, Kids & Toys / Toys / Toddler & Baby\r': 307, 'Jewelry & Watches / Diamond Jewelry / Collections & Sets\r': 414, 'For the Home / Home Appliances / Sewing Machines\r': 428, 'Sports & Outdoors / Fan Shop / NBA\r': 288, 'Baby, Kids & Toys / Health & Safety / Vitamins & Supplements\r': 434, 'Entertainment / Movies & TV / Action & Adventure\r': 5, 'Electronics / Television & Home Theater / Home Theater In Box\r': 406, 'Baby, Kids & Toys / Toys / Games & Puzzles\r': 233, "Women's Fashion / Intimates / Panties\r": 221, 'For the Home / Kitchen & Dining / Food Storage\r': 228, 'Grocery, Household & Pets / Alcohol / Liquor & Spirits\r': 480, "Baby, Kids & Toys / Boys Fashion / Boys' Shoes\r": 438, 'Electronics / Musical Instruments / Microphones & Recording\r': 239, 'Sports & Outdoors / Recreation / Trampolines\r': 492, "Women's Fashion / Shoes / Boat Shoes\r": 452, "Women's Fashion / Accessories / Scarves & Wraps\r": 72, 'For the Home / Bath / Bathrobes\r': 472, 'Grocery, Household & Pets / Household Essentials / Trash Bags\r': 290, "Men's Fashion / Clothing / Swimwear\r": 262, 'Grocery, Household & Pets / Food / Health Foods\r': 397, "Jewelry & Watches / Men's Jewelry / Necklaces\r": 410, 'For the Home / Luggage / Briefcases & Laptop Bags\r': 279, "Jewelry & Watches / Men's Jewelry / Suiting Accessories\r": 364, "Women's Fashion / Accessories / Handbags\r": 175, "Men's Fashion / Shoes / Casual Sneakers\r": 68, 'Grocery, Household & Pets / Candy & Sweets / Hard Candy & Lollipops\r': 278, "Sports & Outdoors / Golf / Men's Golf Clubs\r": 459, 'Grocery, Household & Pets / Tobacco / Cigarettes\r': 385, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Novelty\r': 375, 'For the Home / Mattresses & Accessories / Mattresses\r': 305, "Women's Fashion / Accessories / Umbrellas\r": 333, 'Grocery, Household & Pets / Household Essentials / Laundry\r': 160, 'Sports & Outdoors / Team Sports / Basketball\r': 348, 'Health & Beauty / Personal Care / Deodorants & Antiperspirants\r': 167, "Men's Fashion / Clothing / T-Shirts & Tanks\r": 63, 'Grocery, Household & Pets / Household Essentials / Cleaning Products\r': 490, 'Sports & Outdoors / Fan Shop / MLB\r': 298, 'Jewelry & Watches / Diamond Jewelry / Bracelets & Bangles\r': 413, 'Electronics / Office & School Supplies / Printers & Scanners\r': 75, 'Sports & Outdoors / Exercise & Fitness / Yoga\r': 291, 'For the Home / Kitchen & Dining / Bakeware\r': 363, 'For the Home / Heating & Cooling / Fireplaces\r': 372, 'For the Home / Home Appliances / Stoves & Ranges\r': 219, 'Health & Beauty / Personal Care / Eye Care & Optical\r': 469, 'Electronics / Musical Instruments / Stage Equipment\r': 353, "Women's Fashion / Shoes / Shoe Accessories\r": 360, 'Health & Beauty / Cosmetics / Brushes & Applicators\r': 186, 'For the Home / Patio & Garden / Pools & Water Fun\r': 32, 'Grocery, Household & Pets / Beverages / Juices, Iced Tea & Milk\r': 456, 'Health & Beauty / Hair Care / Styling Tools\r': 96, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Earrings\r': 267, 'Sports & Outdoors / Team Sports / Hockey\r': 316, 'Electronics / Office & School Supplies / Tools & Equipment\r': 304, 'Health & Beauty / Sexual Wellness / Adult Toys For Women\r': 24, 'Health & Beauty / Health Care / Mobility Aids\r': 172, 'Sports & Outdoors / Outdoors / Camping\r': 123, 'Sports & Outdoors / Team Sports / Tennis & Racquet Sports\r': 475, "Women's Fashion / Accessories / Belts\r": 449, 'Electronics / Video Games / Video Game Accessories (529)\r': 338, 'Electronics / Musical Instruments / Other Instruments\r': 355, 'Entertainment / Movies & TV / Kids & Family\r': 214, 'For the Home / Bath / Sink Faucets\r': 461, 'Entertainment / Magazines / Health & Fitness\r': 354, 'Health & Beauty / Cosmetics / Eye Makeup\r': 111, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Collections & Sets\r': 347, 'Entertainment / Books / Self-Improvement\r': 402, 'Electronics / Musical Instruments / Amplifiers & Effects\r': 92, 'For the Home / Patio & Garden / Patio Furniture\r': 141, 'For the Home / Home Decor / Lamps & Lighting\r': 80, 'Electronics / Television & Home Theater / Mounts, Shelves & Consoles\r': 130, "Women's Fashion / Plus Size Clothing / Swimwear\r": 356, 'Health & Beauty / Sexual Wellness / Adult Toys For Men\r': 159, 'Sports & Outdoors / Fan Shop / NCAA\r': 35, "Sports & Outdoors / Golf / Women's Golf Clubs\r": 437, 'Baby, Kids & Toys / Toys / Educational Toys\r': 152, 'Baby, Kids & Toys / Bedding & Bath / Blankets\r': 477, 'For the Home / Patio & Garden / Outdoor Lighting\r': 114, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Necklaces\r': 407, "Jewelry & Watches / Men's Jewelry / Rings\r": 137, 'Baby, Kids & Toys / Baby Care / Feeding\r': 344, "Women's Fashion / Clothing / Jeans\r": 124, 'Auto & Home Improvement / Home Improvement / Electrical\r': 64, "Men's Fashion / Clothing / Underwear & Undershirts\r": 232, 'Auto & Home Improvement / Patio & Garden / Outdoor Storage\r': 440, 'Jewelry & Watches / Gemstone & Pearl Jewelry / Rings\r': 217, 'For the Home / Kitchen & Dining / Gadgets & Utensils\r': 240, 'Entertainment / Books / Children & Young Adult\r': 211, 'Sports & Outdoors / Golf / Accessories\r': 261, 'Electronics / Camera, Video & Surveillance / Camcorders\r': 432, 'Jewelry & Watches / Diamond Jewelry / Earrings\r': 126, 'For the Home / Storage & Organization / Storage Accessories\r': 389, 'Auto & Home Improvement / Automotive / Motorsports\r': 382, 'Baby, Kids & Toys / Toys / Bikes & Ride-ons\r': 323, 'For the Home / Storage & Organization / Garage\r': 120, 'Sports & Outdoors / Fan Shop / NFL\r': 51, 'Electronics / Cell Phones & Accessories / Cell Phones\r': 119, "Women's Fashion / Plus Size Clothing / Activewear\r": 489, 'Electronics / Musical Instruments / DJ Equipment\r': 9, 'Electronics / Camera, Video & Surveillance / Memory & Camera Accessories\r': 36, 'Grocery, Household & Pets / Tobacco / Tobacco Accessories\r': 246, 'Sports & Outdoors / Golf / Golf Shoes\r': 392, 'For the Home / Furniture / Patio & Outdoor Furniture\r': 26, 'Health & Beauty / Vitamins & Supplements / Cleanse & Superfoods\r': 149, 'Health & Beauty / Vitamins & Supplements / Protein\r': 47, "Men's Fashion / Clothing / Suiting & Sport Coats\r": 242, 'For the Home / Home Decor / Pillows & Throws\r': 229, 'Sports & Outdoors / Exercise & Fitness / Strength Training\r': 328, 'For the Home / Kitchen & Dining / Cutlery\r': 115, 'Entertainment / Books / Cookbooks, Food & Wine\r': 238, 'Health & Beauty / Sexual Wellness / Intimate Apparel & Hosiery\r': 125, 'For the Home / Kitchen & Dining / Table Linens & Placemats\r': 33, "Men's Fashion / Shoes / Athletic\r": 255, 'Electronics / Video Games / Games\r': 183, 'Sports & Outdoors / Fan Shop / Premier League\r': 154, 'Entertainment / Magazines / Literature & Writing\r': 450, "Men's Fashion / Accessories / Ties & Bow Ties\r": 308, 'For the Home / Home Decor / \r': 210, "Men's Fashion / Shoes / Boots\r": 357, 'For the Home / Luggage / Travel Accessories\r': 284, 'Health & Beauty / Bath & Body / Body Scrubs & Exfoliants\r': 285, 'Health & Beauty / Sexual Wellness / Sexual Supplements\r': 366, 'Auto & Home Improvement / Automotive / Car Safety & Security\r': 108, 'Health & Beauty / Personal Care / Feminine Care\r': 258, 'Health & Beauty / Health Care / Compression\r': 249, 'Entertainment / Music / R&B\r': 466, 'Sports & Outdoors / Outdoors / Cycling\r': 301, "Men's Fashion / Clothing / Jeans\r": 117, 'Sports & Outdoors / Golf / Golf Clothing\r': 350, 'Health & Beauty / Vitamins & Supplements / Supplements\r': 332, 'For the Home / Bath / Bath Towels\r': 165, 'Electronics / Software / Operating Systems\r': 218, 'Entertainment / Movies & TV / Comedy\r': 69, 'Jewelry & Watches / Fine Metal Jewelry / Collections & Sets\r': 282, 'Entertainment / Movies & TV / Health & Fitness\r': 13, "Women's Fashion / Plus Size Clothing / Tops & Tees\r": 373, 'Electronics / Car Electronics & GPS / Radar Detectors\r': 400, 'For the Home / Bedding / Bed Pillows\r': 235, 'Health & Beauty / Skin Care / Moisturize\r': 91, 'Sports & Outdoors / Outdoors / Boats & Water Sports\r': 222, 'Grocery, Household & Pets / Alcohol / Wine\r': 412, 'Grocery, Household & Pets / Food / Pantry Items\r': 230, 'Sports & Outdoors / Fan Shop / Olympics\r': 423, 'Health & Beauty / Sexual Wellness / Adult Games\r': 295, "Women's Fashion / Intimates / Bras\r": 179, "Men's Fashion / Clothing / Polos\r": 252, 'Electronics / Office & School Supplies / Writing\r': 15, 'For the Home / Storage & Organization / Outdoor\r': 454, 'Auto & Home Improvement / Home Improvement / Plumbing\r': 365, 'Entertainment / Music / Classical\r': 502, "Women's Fashion / Shoes / Oxfords\r": 399, 'Electronics / Television & Home Theater / Set Top Boxes\r': 503, 'Electronics / Software / Production and Editing\r': 497, "Women's Fashion / Plus Size Clothing / Dresses\r": 234, 'Electronics / Computers & Tablets / Tablet Accessories\r': 3, 'Electronics / Cell Phones & Accessories / Backup Batteries\r': 274, 'Electronics / Portable Audio / iPod & MP3 Players\r': 253, 'Grocery, Household & Pets / Food / Baby Foods\r': 73, 'Electronics / Musical Instruments / Brass & Woodwinds\r': 485, 'Auto & Home Improvement / Home Improvement / Batteries\r': 189, "Women's Fashion / Shoes / Pumps & Heels\r": 60, 'Electronics / Musical Instruments / Keyboards & MIDI\r': 465, 'Baby, Kids & Toys / Baby Care / Diapering\r': 421, 'Electronics / Software / Tax\r': 481, 'Jewelry & Watches / Jewelry Accessories & Storage / Cleaners & Accessories\r': 463, "Women's Fashion / Intimates / Shapewear\r": 54, 'Sports & Outdoors / Fan Shop / NHL\r': 309, 'Health & Beauty / Hair Care / Styling Products\r': 4, "Women's Fashion / Accessories / Hats\r": 224, 'For the Home / Luggage / Backpacks\r': 433, 'Electronics / Television & Home Theater / Home Audio\r': 145, "Men's Fashion / Accessories / Sunglasses & Eyewear\r": 18, 'For the Home / Home Decor / Window Treatments\r': 197, 'Sports & Outdoors / Cycling / Parts & Accessories\r': 403, "Men's Fashion / Shoes / Loafers & Slip-Ons\r": 134, 'For the Home / Kitchen & Dining / Cookware\r': 174, 'Grocery, Household & Pets / Candy & Sweets / Fruity, Gummy & Taffy\r': 317, "Women's Fashion / Clothing / Tops & Tees\r": 122, 'Sports & Outdoors / Outdoors / Fishing & Marine\r': 398, 'Health & Beauty / Health Care / Medicine Cabinet\r': 383, "Men's Fashion / Shoes / Boat Shoes\r": 411, 'Health & Beauty / Massage & Relaxation / Foot & Leg Massagers\r': 270, 'Baby, Kids & Toys / Girls Fashion / Jewelry & Watches\r': 374, 'Electronics / Camera, Video & Surveillance / Digital SLRs\r': 426, 'Grocery, Household & Pets / Food / Gourmet Gifts\r': 314, 'Health & Beauty / Cosmetics / Mirrors & Tools\r': 245}
        #cat_dict = cat_dict_list[level-1] # get cat_dict from cat_dict.txt file
        if(level == 1):
            self._classes_dict = dict1
        elif(level == 2):
            self._classes_dict = dict2
        elif(level == 3):
            self._classes_dict = dict3
        #self._classes_dict = dict(cat_dict)
        print self._classes_dict

        for i in range(self._classes_dict.__len__()):
            self._classes_list.append(0)
        for key, value in self._classes_dict.iteritems():
            self._classes_list[value] = key

        self._C_SIZE = len(self._classes_list)

        # set myweight (tf-idf weight matrix)
        #self.myweight = np.zeros((, 11))

        # tf-idf word parsing
        tfidf_file = open("tfidf_" + str(level) + ".txt", "r").read().split('\n')

        for a in range(self._classes_dict.__len__()):
            self.myweight.append({})

        i = 0
        stemmer = SnowballStemmer('english')  # add by Hana
        for line in tfidf_file:
            datas = line.split('||')
            try:
                self.myweight_dic[datas[1]] = i
            except:
                break
            try:
                words = datas[2].split('#')
            except:
                break
            for word in words:
                word_weight = word.split('>')
                try:
                    stemmed_word = stemmer.stem(word_weight[0])
                    self.myweight[i][stemmed_word] = word_weight[1]
                except:
                    break
            i = i + 1

        print("i is done!!!!!!! : ", i)
        print self.myweight_dic

        return

    def getClasses(self):
        return self._classes_list

    #----------------------------------------------------------------------------------------------------
    def setVocab(self, trainingData):   # set feature & make model
        print('> vocab setting start ...')
        index = 0
        clean_item = '' # add by Hana
        stemmed_item = ''

        for (label, line) in trainingData:
            line = self.featureGenerator.getFeatures(line) # change A(upper) to a(lower), split by ' '
            #line_temp = list(set(line)) # remove duplication from line

            # add by Hana
            for item in line:
                if (item not in self._vocab.keys()):
                    self._vocab[item] = index
                    #self._vocab_doc[item] = 1  # add item to _vocab_doc(map)
                    index += 1
                #else:
                    #self._vocab_doc[item] += 1

        self._V_SIZE = len(self._vocab)

        # add by Eunsoo
        for key, value in self._vocab.iteritems():
            self.re_vocab[value] = key

        print('> vocab setting end ...')
        return

    def getVocab(self):
        return self._vocab

    # add by Hana
    #def getVocabDoc(self):
    #    return self._vocab_doc

    def train(self, trainingData):
        pass

    def classify(self, testData, params):
        pass

    def getFeatures(self, data):
        return self.featureGenerator.getFeatures(data)

    def save_classifier(classifier, level, ratio, num_of_training_data):
        f = open('./../../model/160602/160819_model_nb_stem_' + str(level) + '_' + str(ratio) + '_' + str(num_of_training_data) + '.pickle', 'wb')
        pickle.dump(classifier, f, -1)
        f.close()

class FeatureGenerator:
    def getFeatures(self, text):
        stemmer = SnowballStemmer('english')  # add by Hana
        text = text.lower().encode('ascii', 'ignore')  # 160822, ignore unicode @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        texts = text.split()
        feature = []
        for item in texts:
            clean_item = str(filter(str.isalpha, item))  # extract only alpha @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #clean_item = str(item).translate(None, string.punctuation)  # remove '.', ',', ';' ...
            #text = text.translate(None, '0123456789')
            stemmed_item = stemmer.stem(clean_item)
            feature.append(stemmed_item)
        return feature


class NaiveBayesClassifier(Classifier):
    def __init__(self, fg, alpha=0.05): # alpha 0.05 (default)
        Classifier.__init__(self, fg)
        self.__classParams = []
        self.__params = [[]]
        self.__alpha = alpha

    def getParameters(self):
        return (self.__classParams, self.__params)

    def setFeature(self, trainingData, level):
        print('> feature setting start ...')
        self.setClasses(trainingData, level)
        self.setVocab(trainingData)
        self.initParameters()
        #self._vocab_after = self._vocab_doc
        print('> feature setting end ...')

    def train(self, trainingData):
        '''
        self.setClasses(trainingData)
        self.setVocab(trainingData)
        self.initParameters()
        '''

        print('> training start ...')

        for (cat, document) in trainingData:
            for feature in self.getFeatures(document):
                self.countFeature(feature, self._classes_dict[cat])

        for i in range(self._C_SIZE):
            for j in range(self._V_SIZE):
                # print self._counts_in_class[i][j]
                # print self._counts_in_class[i][j] = float(self._counts_in_class[i][j]) * float(1.0 + a)
                tfidf_weight = 0.0
                try:
                    tfidf_weight = self.myweight[self.myweight_dic[self._classes_list[i]]][str(self.re_vocab[j])]
                    print tfidf_weight
                    print "YesYes", self.re_vocab[j], self._classes_list[i]
                except:
                    print "NO", self.re_vocab[j], self._classes_list[i]
                if tfidf_weight > 0:
                    temp = self._counts_in_class[i][j]
                    print type(temp)
                    print type(tfidf_weight)
                    self._counts_in_class[i][j] = float(temp) * (1.0 + float(tfidf_weight)) # add weight

        print('> training end ...')

    def countFeature(self, feature, class_index):
        counts = 1
        self._counts_in_class[class_index][self._vocab[feature]] += counts
        self._total_counts[class_index] += counts
        self._norm += counts

    def classify(self, testData):
        post_prob = self.getPosteriorProbabilities(testData)
        return self._classes_list[self.getMaxIndex(post_prob)]

    def getPosteriorProbabilities(self, testData):
        post_prob = np.zeros(self._C_SIZE)
        for i in range(0, self._C_SIZE):
            for feature in self.getFeatures(testData):
                post_prob[i] += self.getLogProbability(feature, i)
            post_prob[i] += self.getClassLogProbability(i)
        return post_prob

    def getFeatures(self, testData):
        return self.featureGenerator.getFeatures(testData)

    def initParameters(self):
        self._total_counts = np.zeros(self._C_SIZE)
        self._counts_in_class = np.zeros((self._C_SIZE, self._V_SIZE))
        self._norm = 0.0

    def getLogProbability(self, feature, class_index):
        return math.log(self.smooth(self.getCount(feature, class_index), self._total_counts[class_index]))

    def getCount(self, feature, class_index):
        if feature not in self._vocab.keys():
            return 0
        else:
            return self._counts_in_class[class_index][self._vocab[feature]]

    def smooth(self, numerator, denominator):
        return (numerator + self.__alpha) / (denominator + (self.__alpha * len(self._vocab)))

    def getClassLogProbability(self, class_index):
        # add by Eunsoo
        if self._total_counts[class_index] == 0: # add by Eunsoo
            return -1000.0
        else:
            print math.log(self._total_counts[class_index] / self._norm)
            return math.log(self._total_counts[class_index] / self._norm)

    def getMaxIndex(self, posteriorProbabilities):
        maxi = 0
        maxProb = posteriorProbabilities[maxi]
        for i in range(0, self._C_SIZE):
            if (posteriorProbabilities[i] >= maxProb):
                maxProb = posteriorProbabilities[i]
                maxi = i
        return maxi


class Dataset:
    def __init__(self, filename, level, num_of_training_data):
        lines = open(filename, "r").read().decode('utf-8').split('\n')  # add decode('utf-8') 160822 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        num = 0 # total number of data
        wrong = 0
        nutshell = ""
        category = ""
        self.__dataset = []

        for line in lines:
            if num == num_of_training_data: break # number of training data
            else:
                line = lines[num]
                id_nts_ctg = line.split('||') # id || nutshell || category

                # nutshell
                if(id_nts_ctg.__len__() <= 2):
                    #print('> item number is wrong\n')
                    break
                #print(id_nts_ctg[0] + '\n' + id_nts_ctg[1] + '\n' + id_nts_ctg[2] + '\n')
                nutshell = id_nts_ctg[1] # nutshell = '~~~~~~'

                #category
                categories = id_nts_ctg[2].split(" > ") # category = 1 > 2 > 3
                if(categories.__len__() <= 2):
                    #print('> category number is wrong\n')
                    break

                if(level == 1):
                    category = categories[0] # Level 1
                elif(level == 2):
                    category = categories[0] + ' / ' + categories[1] # Level 2
                else:
                    category = categories[0] + ' / ' + categories[1] + ' / ' + categories[2].strip()  # Level 3
                    print categories[2]
                    print categories[2].__len__()

                if(wrong != 1) :
                    self.__dataset.append([category, nutshell])

                num = num + 1
                wrong = 0

        random.shuffle(self.__dataset)
        self.__D_SIZE = num
        self.__trainSIZE = int(0.6 * self.__D_SIZE)
        self.__testSIZE = int(0.3 * self.__D_SIZE)
        self.__devSIZE = 1 - (self.__trainSIZE + self.__testSIZE)

    def setTrainSize(self, value):
        self.__trainSIZE = int(value * 0.01 * self.__D_SIZE)
        return self.__trainSIZE

    def setTestSize(self, value):
        self.__testSIZE = int(value * 0.01 * self.__D_SIZE)
        return self.__testSIZE

    def setDevelopmentSize(self):
        self.__devSIZE = int(1 - (self.__trainSIZE + self.__testSIZE))
        return self.__devSIZE

    def getDataSize(self):
        return self.__D_SIZE

    def getTrainingData(self):
        return self.__dataset[0:self.__trainSIZE]

    def getTestData(self):
        return self.__dataset[self.__trainSIZE:(self.__trainSIZE + self.__testSIZE)]

    def getDevData(self):
        return self.__dataset[0:self.__devSIZE]


# ============================================================================================

if __name__ == "__main__":

    level = int(raw_input('level (1-3) : '))
    ratio = str(raw_input('ratio of training:test (82, 73, 64, 55) : '))
    num_of_training_data = int(raw_input('number of training data : '))
    print('\n')

    infile = "./../../../data/160819_name.txt"
    outfile = open("test.txt", 'w')

    if len(sys.argv) > 1:
        infile = sys.argv[1]

    data = Dataset(infile, level, num_of_training_data)

    ratio_training = int(ratio[0] + '0')
    ratio_test = int(ratio[1] + '0')
    data.setTrainSize(ratio_training)
    data.setTestSize(ratio_test)

    train_set = data.getTrainingData()
    test_set = data.getTestData()

    test_data = [test_set[i][1] for i in range(len(test_set))]
    actual_labels = [test_set[i][0] for i in range(len(test_set))]

    fg = FeatureGenerator()
    alpha = 0.5  # smoothing parameter

    nbClassifier = NaiveBayesClassifier(fg, alpha)

    # training start
    now1 = datetime.now()

    nbClassifier.setFeature(train_set, level)
    nbClassifier.train(train_set)

    #nbClassifier._vocab.pop('i')
    #nbClassifier._vocab.pop('is')
    #nbClassifier._vocab.pop('the')

    now2 = datetime.now()
    # training end

    training_time = now2 - now1

    nbClassifier.save_classifier(level, ratio, num_of_training_data)

    # accuracy test
    print('\n> test start ...\n')
    outfile.write('\n> test start ...\n')
    correct = 0
    total = 0

    cat_dict = nbClassifier._classes_dict
    num = cat_dict.__len__()
    eunsoo = np.zeros((num, num))

    for line in test_data:
        line = line.lower().encode('ascii', 'ignore')

        best_label = nbClassifier.classify(line)
        # print(str(total) + '. ' + line + '\n\t' + best_label + ' =?= ' + actual_labels[total])
        # outfile.write('\n' + str(total) + '. ' + line + '\n\t' + best_label + ' =?= ' + actual_labels[total])
        if best_label == actual_labels[total]:
            correct += 1
            # print('O')
            # outfile.write(' -> O')
        else:
            # print('X')
            # outfile.write(' -> X')

            print(str(total) + '. ' + str(line) + '\n\t' + str(best_label) + ' =/= ' + str(actual_labels[total]))
            outfile.write('\n' + str(total) + '. ' + str(line) + '\n\t' + str(best_label) + ' =/= ' + str(actual_labels[total]))

            # check eunsoo
            #print actual_labels[total]
            #print type(str(actual_labels[total]))
            '''if cat_dict.has_key(actual_labels[total]) != True: # There is no test category in training category(cat_dict)
                cat_dict[actual_labels[total]] = num - 1
                num += 1
                print 'num : ' + str(num)
                for i in range(initial_num): # add row in eunsoo
                    eunsoo[num - 1][i] = 0
                print eunsoo.shape()
'''
            actual_label = cat_dict[actual_labels[total]]
            print 'act index : ' + str(actual_label)
            wrong_label = cat_dict[best_label]
            print 'wrong index : ' + str(wrong_label)
            eunsoo[actual_label][wrong_label] += 1

        total += 1

    acc = 1.0 * correct / total

    print('\n> test end ...\n')
    outfile.write('\n> test end ...\n')
    print('=' * 60)
    outfile.write('=' * 60)
    print(' RESULT')
    outfile.write('\n\n RESULT')
    print('=' * 60)
    outfile.write('\n')
    outfile.write('=' * 60)
    print(' - Level : ' + str(level))
    outfile.write('\n - Level : ' + str(level))
    print(' - Ratio of training:test : ' + str(ratio_training) + ':' + str(ratio_test))
    outfile.write('\n - Ratio of training:test : ' + str(ratio_training) + ':' + str(ratio_test))
    print(' - Amount of data : ' + str(num_of_training_data))
    outfile.write('\n - Amount of data : ' + str(num_of_training_data))
    temp1 = ' - Training time : %d' % training_time.total_seconds()
    print(temp1)
    outfile.write('\n' + temp1)
    temp2 = ' - Accuracy : %0.3f' % acc
    print(temp2)
    outfile.write('\n' + temp2)
    print(' - Amount of Category : ' + str(nbClassifier.getClasses().__len__()))
    outfile.write('\n - Amount of Category : ' + str(nbClassifier.getClasses().__len__()))
    print('=' * 60)
    outfile.write('\n')
    outfile.write('=' * 60)

    print('\n' + str(cat_dict))
    outfile.write('\n' + str(cat_dict))
    print('\n')
    print(eunsoo)
    outfile.write('\n' + str(eunsoo))

    outfile.close()



