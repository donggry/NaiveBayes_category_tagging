# coding: utf-8
from __future__ import print_function
import numpy as np
from operator import itemgetter

import string
import numpy as np
import textmining
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords
wnl = WordNetLemmatizer()

StopWords = set(stopwords.words("english"))
StopWords.update(('and', 'may', 'a', 'use', 'us','so', 'your', 'this', 'when', 'it', 'many', 'can', 'set', 'cant',
                            '’','®','™','®','with','º','in','to','v', 'w\xc2\xb0','yes', 'not', 'no', 'these','keep','enough','use','x','w','d','p','n'))

g=open("tfidf22.txt","w")
tro=open("tfidf33.txt","w")
vocab=[]

doc_dic=[]
for a in range(508):###################################small category change!!!!!!!!
    doc_dic.append([])
category_dic = {'Health & Beauty / Cosmetics / Face Makeup': 250, "Men's Fashion / Shoes / Sandals": 225,
                      'Jewelry & Watches / Fashion Jewelry / Collections & Sets': 318,
                      'For the Home / Bedding / Duvet Covers': 21, "Women's Fashion / Clothing / Pants": 180,
                      'For the Home / Home Decor / Home Accents': 319,
                      'Jewelry & Watches / Diamond Jewelry / Necklaces': 45,
                      'Electronics / Office & School Supplies / Printer Ink & Toner': 170,
                      "Women's Fashion / Clothing / Costumes": 93,
                      'Grocery, Household & Pets / Food / Snack Foods': 110,
                      'Sports & Outdoors / Team Sports / Soccer': 293,
                      'Sports & Outdoors / Team Sports / Baseball & Softball': 424,
                      'Health & Beauty / Sexual Wellness / Bondage & Fetish': 321,
                      'For the Home / Home Appliances / Small Appliances': 129,
                      'For the Home / Art / Framed Art': 49,
                      'Sports & Outdoors / Cycling / Clothing & Footwear': 448,
                      'Electronics / Portable Audio / In-Ear & Earbud Headphones': 39,
                      'Auto & Home Improvement / Home Appliances / Dishwashers': 462,
                      'Auto & Home Improvement / Home Improvement / Tool Storage': 151,
                      "Men's Fashion / Shoes / Oxfords": 161,
                      'Electronics / Cell Phones & Accessories / Screen Protectors': 247,
                      'Electronics / Computers & Tablets / Computer Accessories': 25,
                      'Electronics / Musical Instruments / Drums & Percussion': 248,
                      'Electronics / Computers & Tablets / Laptops': 181,
                      'Auto & Home Improvement / Home Improvement / Heating & Cooling': 341,
                      'Health & Beauty / Health Care / Humidifiers': 203,
                      'Electronics / Portable Audio / On-Ear & Over-Ear Headphones': 280,
                      'Health & Beauty / Health Care / Medical Braces': 312,
                      'Jewelry & Watches / Fine Metal Jewelry / Bracelets & Bangles': 162,
                      "Men's Fashion / Clothing / Pants": 205, 'Jewelry & Watches / Diamond Jewelry / Novelty': 491,
                      'Electronics / Cell Phones & Accessories / Cables, Chargers & Adapters': 81,
                      'For the Home / Home Decor / Candles & Holders': 131,
                      'Sports & Outdoors / Team Sports / Lacrosse': 259, 'For the Home / Art / Photography': 185,
                      'Electronics / Musical Instruments / Guitars': 202,
                      'Entertainment / Movies & TV / Television': 12,
                      'For the Home / Patio & Garden / Bird Feeders & Food': 371,
                      'Baby, Kids & Toys / Toys / Pretend Play': 286,
                      'Jewelry & Watches / Diamond Jewelry / Rings': 349,
                      'Electronics / Software / Business & Home Office': 417,
                      'For the Home / Bedding / Down & Alternative Comforters': 343,
                      'Grocery, Household & Pets / Candy & Sweets / Fruit & Nut': 495,
                      'Auto & Home Improvement / Home Improvement / Power Tools': 164,
                      "Women's Fashion / Clothing / Activewear": 143,
                      'For the Home / Home Appliances / Vacuums & Floor Care': 67,
                      'Jewelry & Watches / Fashion Jewelry / Necklaces': 379,
                      'Health & Beauty / Massage & Relaxation / Handheld Massagers': 118,
                      "Men's Fashion / Clothing / Shorts": 65, 'Baby, Kids & Toys / Toys / Outdoor Play': 155,
                      'Electronics / Portable Audio / Bluetooth & Wireless Speakers': 191,
                      "Men's Fashion / Clothing / Socks": 272, 'Health & Beauty / Personal Care / Oral Care': 135,
                      'Jewelry & Watches / Fashion Jewelry / Novelty': 335,
                      'Health & Beauty / Sexual Wellness / Adult Toys for Couples': 387,
                      'Baby, Kids & Toys / Toys / Dolls & Action Figures': 169,
                      "Men's Fashion / Accessories / Wallets & Money Clips": 57, 'Entertainment / Music / Rap': 474,
                      'Entertainment / Movies & TV / History & Documentary': 192,
                      'Health & Beauty / Skin Care / Hair Removal': 395,
                      'For the Home / Home Appliances / Irons & Garment Care': 43,
                      'Health & Beauty / Personal Care / Shaving & Grooming': 11,
                      'Electronics / Musical Instruments / Accessories': 153,
                      "Men's Fashion / Clothing / Activewear": 176,
                      'For the Home / Heating & Cooling / Dehumidifiers': 370,
                      "Women's Fashion / Clothing / Outerwear & Suiting": 46,
                      'Electronics / Television & Home Theater / Blu Ray & DVD Players': 231,
                      "Baby, Kids & Toys / Girls Fashion / Girls' Shoes": 116,
                      'Jewelry & Watches / Fine Metal Jewelry / Necklaces': 306,
                      "Men's Fashion / Shoes / Shoe Accessories": 443,
                      "Jewelry & Watches / Watches / Men's Watches": 296,
                      'For the Home / Bath / Bath Accessories & Sets': 327,
                      'Electronics / Car Electronics & GPS / Car Security & Remote Start': 460,
                      "Jewelry & Watches / Men's Jewelry / Novelty": 394,
                      'Electronics / Software / Programming': 457,
                      'For the Home / Patio & Garden / Saunas Spas & Hot Tubs': 471,
                      "Men's Fashion / Accessories / Bags": 37,
                      'Health & Beauty / Vitamins & Supplements / Weight Loss': 20,
                      'Auto & Home Improvement / Home Improvement / Garage': 345,
                      'Electronics / Camera, Video & Surveillance / Point & Shoots': 273,
                      'Entertainment / Video Games / Game Gear & Novelties': 455,
                      "Women's Fashion / Shoes / Evening": 103, 'For the Home / Luggage / Hardside': 388,
                      'Baby, Kids & Toys / Maternity / Tops': 6,
                      'Health & Beauty / Health Care / Healthy Living': 53,
                      'For the Home / Patio & Garden / Gardening & Lawn Care': 95,
                      'Grocery, Household & Pets / Candy & Sweets / Bakery': 206,
                      'Entertainment / Magazines / Business & Finance': 507,
                      "Women's Fashion / Clothing / Sweaters & Cardigans": 87,
                      'Baby, Kids & Toys / Boys Fashion / Watches': 204,
                      'For the Home / Furniture / Accent Furniture': 376,
                      'Health & Beauty / Vitamins & Supplements / Vitamins': 283,
                      'For the Home / Home Decor / Picture Frames': 322,
                      "Women's Fashion / Intimates / Socks & Hosiery": 19,
                      'Health & Beauty / Cosmetics / Lips': 142,
                      "Women's Fashion / Plus Size Clothing / Intimates": 324,
                      'Electronics / Television & Home Theater / TVs': 190,
                      'Auto & Home Improvement / Automotive / Exterior Accessories': 62,
                      'Health & Beauty / Massage & Relaxation / Head Massagers': 418,
                      "Women's Fashion / Accessories / Gloves": 257, 'Entertainment / Books / Hobbies': 337,
                      'Jewelry & Watches / Fashion Jewelry / Bracelets & Bangles': 223,
                      'Auto & Home Improvement / Automotive / Car Care': 302,
                      'Health & Beauty / Skin Care / Sun Care & Tanning': 188,
                      'For the Home / Home Appliances / Wine Coolers': 420,
                      'Sports & Outdoors / Exercise & Fitness / Cardio Training': 163,
                      'For the Home / Storage & Organization / Bathroom': 369,
                      "For the Home / Furniture / Baby & Kid's Furniture": 98,
                      'Grocery, Household & Pets / Tobacco / Vaporizers & E-Cigs': 256,
                      'Entertainment / Movies & TV / Sci-fi & Fantasy': 14,
                      "Women's Fashion / Maternity Clothing / Intimates": 444,
                      'Sports & Outdoors / Exercise & Fitness / Books & Magazines': 427,
                      "Women's Fashion / Clothing / Skirts": 216,
                      'For the Home / Furniture / Kitchen & Dining Furniture': 100,
                      "Women's Fashion / Accessories / Sunglasses & Eyewear": 85,
                      'Health & Beauty / Vitamins & Supplements / Multi & Prenatal Vitamins': 127,
                      'For the Home / Bedding / Blankets & Throws': 173,
                      'Grocery, Household & Pets / Beverages / Sports Drinks': 446,
                      'Electronics / Office & School Supplies / Paper & Stationery': 254,
                      "Men's Fashion / Shoes / Dress Shoes": 182,
                      "Women's Fashion / Shoes / Loafers & Slip-Ons": 264,
                      'Jewelry & Watches / Fine Metal Jewelry / Novelty': 367,
                      'Electronics / Office & School Supplies / School Supplies': 76,
                      'Grocery, Household & Pets / Household Essentials / Food Storage': 22,
                      'Sports & Outdoors / Exercise & Fitness / Balance & Recovery': 435,
                      'Grocery, Household & Pets / Beverages / Water': 473,
                      'For the Home / Kitchen & Dining / Kitchen Appliances': 493,
                      'Health & Beauty / Bath & Body / Accessories': 0,
                      'Jewelry & Watches / Diamond Jewelry / Wedding & Engagement': 276,
                      'For the Home / Home Decor / Rugs': 132,
                      'Health & Beauty / Skin Care / Treatments & Serums': 42,
                      "Baby, Kids & Toys / Girls Fashion / Girls' Clothing": 121,
                      'Health & Beauty / Sexual Wellness / Lubricants & Sexual Enhancers': 196,
                      'Auto & Home Improvement / Home Improvement / Bathroom Faucets': 506,
                      'Health & Beauty / Health Care / Health Monitors': 112,
                      'For the Home / Patio & Garden / Outdoor Power Equipment': 220,
                      'For the Home / Luggage / Carry-Ons': 334, 'Baby, Kids & Toys / Toys / Arts & Crafts': 146,
                      "Men's Fashion / Clothing / Outerwear & Jackets": 56,
                      'Jewelry & Watches / Watches / Watch Accessories': 168,
                      'For the Home / Art / Prints & Decals': 140,
                      "Women's Fashion / Clothing / Shorts & Capris": 90,
                      'Health & Beauty / Health Care / Pain Relief': 208,
                      'Health & Beauty / Sexual Wellness / Anal Toys': 187,
                      'Health & Beauty / Health Care / First Aid': 380,
                      "Men's Fashion / Clothing / Lounge & Sleepwear": 89, 'Entertainment / Music / Rock': 396,
                      "Baby, Kids & Toys / Boys Fashion / Boys' Accessories": 243,
                      'For the Home / Bedding / Sheets': 34,
                      'Electronics / Television & Home Theater / Home Theater Accessories': 8,
                      'Electronics / Computers & Tablets / Desktops, Monitors, & All-In-Ones': 244,
                      'For the Home / Storage & Organization / Laundry': 166,
                      'Health & Beauty / Massage & Relaxation / Acupuncture & Acupressure': 404,
                      "Jewelry & Watches / Men's Jewelry / Earrings": 269,
                      "For the Home / Luggage / Kid's Travel Bags": 177, 'For the Home / Art / Canvas': 16,
                      'Health & Beauty / Bath & Body / Hands & Feet': 391,
                      'Grocery, Household & Pets / Household Essentials / Home Fragrance & Air Care': 441,
                      "Men's Fashion / Accessories / Pocket Squares": 482,
                      'For the Home / Patio & Garden / Grills & Outdoor Cooking': 292,
                      'Grocery, Household & Pets / Beverages / Coffee': 97,
                      'Electronics / Cell Phones & Accessories / Bluetooth Devices': 227,
                      'For the Home / Mattresses & Accessories / Memory Foam Mattresses': 71,
                      'Health & Beauty / Personal Care / Incontinence': 467,
                      'Grocery, Household & Pets / Candy & Sweets / Chocolate': 464,
                      'For the Home / Furniture / Living Room Furniture': 28,
                      'For the Home / Storage & Organization / Office': 326,
                      'Grocery, Household & Pets / Alcohol / Beer': 500,
                      "Jewelry & Watches / Men's Jewelry / Collections & Sets": 430,
                      'Sports & Outdoors / Recreation / Lawn Games': 378,
                      "Sports & Outdoors / Clothing & Shoes / Men's Activewear": 213,
                      'Baby, Kids & Toys / Health & Safety / Baby Monitors': 468,
                      "Entertainment / Music / Kid's Music": 504,
                      'Health & Beauty / Massage & Relaxation / Total Body Massagers': 241,
                      "Health & Beauty / Fragrance / Men's Fragrance": 66,
                      'Health & Beauty / Hair Care / Hair Accessories': 157,
                      'Electronics / Portable Audio / Docks, Radios & Boom Boxes': 83,
                      'Sports & Outdoors / Outdoors / Action Sports': 77,
                      "Women's Fashion / Clothing / Swimwear": 195,
                      "Baby, Kids & Toys / Girls Fashion / Girls' Accessories": 346,
                      'Grocery, Household & Pets / Candy & Sweets / Gum & Mints': 156,
                      'For the Home / Kitchen & Dining / Serveware': 212,
                      'Grocery, Household & Pets / Alcohol / Mixers & Ready To Drink': 488,
                      'Electronics / Office & School Supplies / Home Office Furniture': 199,
                      'Baby, Kids & Toys / Toys / Electronic Toys': 265,
                      "Baby, Kids & Toys / Boys Fashion / Boys' Clothing": 263,
                      'Health & Beauty / Personal Care / Foot Care': 147,
                      'Auto & Home Improvement / Automotive / Interior Accessories': 84,
                      'For the Home / Storage & Organization / Closet': 41,
                      'Electronics / Office & School Supplies / Shredders': 386,
                      "Women's Fashion / Intimates / Lounge & Sleepwear": 139,
                      'Auto & Home Improvement / Home Improvement / Home Safety & Security': 17,
                      'Auto & Home Improvement / Home Improvement / Hardware': 277,
                      'For the Home / Home Appliances / Dishwashers': 470,
                      "Men's Fashion / Accessories / Scarves, Hats & Gloves": 79,
                      'Baby, Kids & Toys / Toys / Building Sets & Blocks': 55,
                      'Health & Beauty / Massage & Relaxation / Massage Oils, Aromatherapy & Lotions': 44,
                      'Grocery, Household & Pets / Beverages / Powdered Drink Mixes': 416,
                      'For the Home / Kitchen & Dining / Dinnerware': 136,
                      'For the Home / Luggage / Duffel Bags': 260, "Women's Fashion / Accessories / Wallets": 207,
                      'For the Home / Kitchen & Dining / Flatware': 431,
                      'Sports & Outdoors / Golf / Golf Balls': 342,
                      'Jewelry & Watches / Fine Metal Jewelry / Earrings': 422,
                      "Women's Fashion / Plus Size Clothing / Bottoms": 193,
                      'Entertainment / Music / Soundtrack': 479,
                      'Health & Beauty / Bath & Body / Body Cleansers': 109,
                      'Electronics / Office & School Supplies / Packing & Mailing': 237,
                      'Sports & Outdoors / Golf / Golf Bags and Cart': 405,
                      'Health & Beauty / Skin Care / Cleanse': 40,
                      'Electronics / Television & Home Theater / Projectors & Screens': 133,
                      'For the Home / Bath / Bath Rugs': 251, 'Health & Beauty / Cosmetics / Nails': 128,
                      'Health & Beauty / Cosmetics / Makeup Palettes & Sets': 27,
                      'Sports & Outdoors / Fan Shop / MLS': 289, "Women's Fashion / Clothing / Dresses": 106,
                      'Electronics / Office & School Supplies / Scanners': 408,
                      'Health & Beauty / Massage & Relaxation / Massage Accessories': 88,
                      'Baby, Kids & Toys / Bedding & Bath / Crib Sets': 381,
                      'Baby, Kids & Toys / Bedding & Bath / Potty Training': 496,
                      "Men's Fashion / Shoes / Slippers": 201, 'Health & Beauty / Cosmetics / Bags & Cases': 393,
                      'Health & Beauty / Health Care / Daily Living Aids': 236,
                      'Entertainment / Movies & TV / Drama': 2,
                      'Auto & Home Improvement / Home Appliances / Vacuums & Floor Care': 442,
                      'Electronics / Computers & Tablets / Tablets & E-Readers': 78,
                      'Auto & Home Improvement / Home Improvement / Paint & Wallpapers': 194,
                      "Women's Fashion / Clothing / Leggings": 7, 'Entertainment / Music / Country': 501,
                      'For the Home / Bath / Beach Towels': 359, 'For the Home / Bedding / Comforters': 86,
                      "Jewelry & Watches / Watches / Women's Watches": 451,
                      'Sports & Outdoors / Team Sports / Football': 415,
                      'Health & Beauty / Personal Care / Body Treatments': 99,
                      'Baby, Kids & Toys / Health & Safety / Safety': 74,
                      'Jewelry & Watches / Fashion Jewelry / Earrings': 275,
                      "Women's Fashion / Shoes / Athletic": 300,
                      'Electronics / Office & School Supplies / Organization': 144,
                      'Health & Beauty / Bath & Body / Bath Soaks & Bubble Baths': 38,
                      'Baby, Kids & Toys / Maternity / Bottoms': 311,
                      'For the Home / Home Appliances / Washers & Dryers': 425,
                      "Men's Fashion / Clothing / Costumes": 266, 'For the Home / Luggage / Luggage Sets': 226,
                      'Grocery, Household & Pets / Beverages / Coconut Water': 377,
                      "Women's Fashion / Shoes / Fashion Sneakers": 148, 'Entertainment / Music / Pop': 445,
                      'Electronics / Cell Phones & Accessories / Cases': 61,
                      'Grocery, Household & Pets / Beverages / Hot Cocoa': 498,
                      'For the Home / Kitchen & Dining / Kitchen Towels & Aprons': 429,
                      'For the Home / Kitchen & Dining / Drinkware': 29,
                      'Grocery, Household & Pets / Beverages / Tea': 107,
                      'Entertainment / Magazines / Hobbies & Crafts': 499,
                      'Entertainment / Video Games / Game Consoles': 268,
                      'Grocery, Household & Pets / Pets / Cats': 138,
                      "Women's Fashion / Plus Size Clothing / Outerwear & Suiting": 113,
                      'For the Home / Bedding / Baby Bedding': 271, 'Baby, Kids & Toys / Maternity / Nursing': 484,
                      'Electronics / Cell Phones & Accessories / Cell Phone Accessories': 82,
                      'For the Home / Furniture / Bedroom Furniture': 10,
                      'Grocery, Household & Pets / Food / Breakfast Foods': 294,
                      'For the Home / Bath / Bathroom Scales': 362,
                      'For the Home / Bath / Shower Curtains & Liners': 303,
                      'Sports & Outdoors / Team Sports / Volleyball': 494,
                      'Health & Beauty / Health Care / Sleep Aids': 336,
                      'Jewelry & Watches / Fashion Jewelry / Rings': 352,
                      'Baby, Kids & Toys / Bedding & Bath / Bath Tubs & Seats': 281,
                      'Health & Beauty / Hair Care / Shampoo & Conditioner': 23,
                      'Baby, Kids & Toys / Baby Care / Baby Gear': 325, "Women's Fashion / Shoes / Boots": 1,
                      'Electronics / Camera, Video & Surveillance / Security & Surveillance': 52,
                      "Women's Fashion / Shoes / Sandals": 59, 'Entertainment / Books / Non-Fiction': 436,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Bracelets & Bangles': 329,
                      'For the Home / Furniture / Bathroom Furniture': 505,
                      'For the Home / Home Appliances / Refrigerators': 483,
                      "Women's Fashion / Shoes / Slippers": 200,
                      'Jewelry & Watches / Jewelry Accessories & Storage / Boxes & Holders': 339,
                      'Electronics / Video Games / Video Game Accessories': 184,
                      'Health & Beauty / Bath & Body / Body Moisturizers': 297,
                      "Men's Fashion / Clothing / Sweaters & Sweatshirts": 48,
                      'Baby, Kids & Toys / Maternity / Dresses': 340,
                      'For the Home / Bedding / Mattress Toppers & Pads': 361,
                      'Electronics / Camera, Video & Surveillance / Action Cameras & Drones': 171,
                      'Grocery, Household & Pets / Beverages / Soft Drinks': 476,
                      'Electronics / Video Games / Games (92)': 299, "Women's Fashion / Intimates / Lingerie": 105,
                      'Sports & Outdoors / Fan Shop / Memorabilia': 478,
                      'Sports & Outdoors / Fan Shop / NASCAR': 487,
                      "Men's Fashion / Accessories / Belts & Suspenders": 287,
                      'Health & Beauty / Hair Care / Hair Color': 158,
                      'Auto & Home Improvement / Home Improvement / Lighting': 150,
                      'Electronics / Cell Phones & Accessories / Mounts & Stands': 419,
                      'Sports & Outdoors / Outdoors / Hunting': 351,
                      'Grocery, Household & Pets / Candy & Sweets / Assortments': 439,
                      'Entertainment / Video Games / Game Consoles (53)': 458,
                      'Auto & Home Improvement / Home Improvement / Hand Tools': 104,
                      "Women's Fashion / Shoes / Flats": 50, 'Grocery, Household & Pets / Pets / Dogs': 101,
                      'Grocery, Household & Pets / Pets / Fish': 447,
                      'Grocery, Household & Pets / Tobacco / Cigars': 486,
                      'Jewelry & Watches / Fine Metal Jewelry / Rings': 331,
                      'For the Home / Luggage / Suitcases': 215,
                      'Auto & Home Improvement / Automotive / Car Electronics': 94,
                      'For the Home / Furniture / Home Office Furniture': 30,
                      'Electronics / Television & Home Theater / Streaming Media Players & Antennas': 368,
                      'Health & Beauty / Massage & Relaxation / Pulse Massagers': 209,
                      'Grocery, Household & Pets / Pets / Birds': 409,
                      'For the Home / Bath / Bath Storage & Caddies': 310,
                      'Grocery, Household & Pets / Pets / Small Animals': 358,
                      'Sports & Outdoors / Exercise & Fitness / Fitness Technology': 313,
                      'Auto & Home Improvement / Home Improvement / Flooring': 330,
                      'Entertainment / Books / Fiction & Literature': 390,
                      'Electronics / Computers & Tablets / Networking & Wireless': 401,
                      "Health & Beauty / Fragrance / Women's Fragrance": 70,
                      'Health & Beauty / Personal Care / Pregnancy & Fertility': 178,
                      'Sports & Outdoors / Outdoors / Winter Sports': 384,
                      'Health & Beauty / Vitamins & Supplements / Sports Nutrition': 31,
                      'For the Home / Bedding / Quilts & Bedspreads': 198,
                      "Jewelry & Watches / Men's Jewelry / Bracelets": 320,
                      'Auto & Home Improvement / Home Improvement / Shower Heads': 315,
                      'Electronics / Portable Audio / Headphone Accessories': 453,
                      'Health & Beauty / Vitamins & Supplements / Herbal Remedies & Teas': 102,
                      "Men's Fashion / Clothing / Button-Down Shirts": 58,
                      'Baby, Kids & Toys / Toys / Toddler & Baby': 307,
                      'Jewelry & Watches / Diamond Jewelry / Collections & Sets': 414,
                      'For the Home / Home Appliances / Sewing Machines': 428,
                      'Sports & Outdoors / Fan Shop / NBA': 288,
                      'Baby, Kids & Toys / Health & Safety / Vitamins & Supplements': 434,
                      'Entertainment / Movies & TV / Action & Adventure': 5,
                      'Electronics / Television & Home Theater / Home Theater In Box': 406,
                      'Baby, Kids & Toys / Toys / Games & Puzzles': 233,
                      "Women's Fashion / Intimates / Panties": 221,
                      'For the Home / Kitchen & Dining / Food Storage': 228,
                      'Grocery, Household & Pets / Alcohol / Liquor & Spirits': 480,
                      "Baby, Kids & Toys / Boys Fashion / Boys' Shoes": 438,
                      'Electronics / Musical Instruments / Microphones & Recording': 239,
                      'Sports & Outdoors / Recreation / Trampolines': 492,
                      "Women's Fashion / Shoes / Boat Shoes": 452,
                      "Women's Fashion / Accessories / Scarves & Wraps": 72, 'For the Home / Bath / Bathrobes': 472,
                      'Grocery, Household & Pets / Household Essentials / Trash Bags': 290,
                      "Men's Fashion / Clothing / Swimwear": 262,
                      'Grocery, Household & Pets / Food / Health Foods': 397,
                      "Jewelry & Watches / Men's Jewelry / Necklaces": 410,
                      'For the Home / Luggage / Briefcases & Laptop Bags': 279,
                      "Jewelry & Watches / Men's Jewelry / Suiting Accessories": 364,
                      "Women's Fashion / Accessories / Handbags": 175,
                      "Men's Fashion / Shoes / Casual Sneakers": 68,
                      'Grocery, Household & Pets / Candy & Sweets / Hard Candy & Lollipops': 278,
                      "Sports & Outdoors / Golf / Men's Golf Clubs": 459,
                      'Grocery, Household & Pets / Tobacco / Cigarettes': 385,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Novelty': 375,
                      'For the Home / Mattresses & Accessories / Mattresses': 305,
                      "Women's Fashion / Accessories / Umbrellas": 333,
                      'Grocery, Household & Pets / Household Essentials / Laundry': 160,
                      'Sports & Outdoors / Team Sports / Basketball': 348,
                      'Health & Beauty / Personal Care / Deodorants & Antiperspirants': 167,
                      "Men's Fashion / Clothing / T-Shirts & Tanks": 63,
                      'Grocery, Household & Pets / Household Essentials / Cleaning Products': 490,
                      'Sports & Outdoors / Fan Shop / MLB': 298,
                      'Jewelry & Watches / Diamond Jewelry / Bracelets & Bangles': 413,
                      'Electronics / Office & School Supplies / Printers & Scanners': 75,
                      'Sports & Outdoors / Exercise & Fitness / Yoga': 291,
                      'For the Home / Kitchen & Dining / Bakeware': 363,
                      'For the Home / Heating & Cooling / Fireplaces': 372,
                      'For the Home / Home Appliances / Stoves & Ranges': 219,
                      'Health & Beauty / Personal Care / Eye Care & Optical': 469,
                      'Electronics / Musical Instruments / Stage Equipment': 353,
                      "Women's Fashion / Shoes / Shoe Accessories": 360,
                      'Health & Beauty / Cosmetics / Brushes & Applicators': 186,
                      'For the Home / Patio & Garden / Pools & Water Fun': 32,
                      'Grocery, Household & Pets / Beverages / Juices, Iced Tea & Milk': 456,
                      'Health & Beauty / Hair Care / Styling Tools': 96,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Earrings': 267,
                      'Sports & Outdoors / Team Sports / Hockey': 316,
                      'Electronics / Office & School Supplies / Tools & Equipment': 304,
                      'Health & Beauty / Sexual Wellness / Adult Toys For Women': 24,
                      'Health & Beauty / Health Care / Mobility Aids': 172,
                      'Sports & Outdoors / Outdoors / Camping': 123,
                      'Sports & Outdoors / Team Sports / Tennis & Racquet Sports': 475,
                      "Women's Fashion / Accessories / Belts": 449,
                      'Electronics / Video Games / Video Game Accessories (529)': 338,
                      'Electronics / Musical Instruments / Other Instruments': 355,
                      'Entertainment / Movies & TV / Kids & Family': 214, 'For the Home / Bath / Sink Faucets': 461,
                      'Entertainment / Magazines / Health & Fitness': 354,
                      'Health & Beauty / Cosmetics / Eye Makeup': 111,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Collections & Sets': 347,
                      'Entertainment / Books / Self-Improvement': 402,
                      'Electronics / Musical Instruments / Amplifiers & Effects': 92,
                      'For the Home / Patio & Garden / Patio Furniture': 141,
                      'For the Home / Home Decor / Lamps & Lighting': 80,
                      'Electronics / Television & Home Theater / Mounts, Shelves & Consoles': 130,
                      "Women's Fashion / Plus Size Clothing / Swimwear": 356,
                      'Health & Beauty / Sexual Wellness / Adult Toys For Men': 159,
                      'Sports & Outdoors / Fan Shop / NCAA': 35,
                      "Sports & Outdoors / Golf / Women's Golf Clubs": 437,
                      'Baby, Kids & Toys / Toys / Educational Toys': 152,
                      'Baby, Kids & Toys / Bedding & Bath / Blankets': 477,
                      'For the Home / Patio & Garden / Outdoor Lighting': 114,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Necklaces': 407,
                      "Jewelry & Watches / Men's Jewelry / Rings": 137,
                      'Baby, Kids & Toys / Baby Care / Feeding': 344, "Women's Fashion / Clothing / Jeans": 124,
                      'Auto & Home Improvement / Home Improvement / Electrical': 64,
                      "Men's Fashion / Clothing / Underwear & Undershirts": 232,
                      'Auto & Home Improvement / Patio & Garden / Outdoor Storage': 440,
                      'Jewelry & Watches / Gemstone & Pearl Jewelry / Rings': 217,
                      'For the Home / Kitchen & Dining / Gadgets & Utensils': 240,
                      'Entertainment / Books / Children & Young Adult': 211,
                      'Sports & Outdoors / Golf / Accessories': 261,
                      'Electronics / Camera, Video & Surveillance / Camcorders': 432,
                      'Jewelry & Watches / Diamond Jewelry / Earrings': 126,
                      'For the Home / Storage & Organization / Storage Accessories': 389,
                      'Auto & Home Improvement / Automotive / Motorsports': 382,
                      'Baby, Kids & Toys / Toys / Bikes & Ride-ons': 323,
                      'For the Home / Storage & Organization / Garage': 120,
                      'Sports & Outdoors / Fan Shop / NFL': 51,
                      'Electronics / Cell Phones & Accessories / Cell Phones': 119,
                      "Women's Fashion / Plus Size Clothing / Activewear": 489,
                      'Electronics / Musical Instruments / DJ Equipment': 9,
                      'Electronics / Camera, Video & Surveillance / Memory & Camera Accessories': 36,
                      'Grocery, Household & Pets / Tobacco / Tobacco Accessories': 246,
                      'Sports & Outdoors / Golf / Golf Shoes': 392,
                      'For the Home / Furniture / Patio & Outdoor Furniture': 26,
                      'Health & Beauty / Vitamins & Supplements / Cleanse & Superfoods': 149,
                      'Health & Beauty / Vitamins & Supplements / Protein': 47,
                      "Men's Fashion / Clothing / Suiting & Sport Coats": 242,
                      'For the Home / Home Decor / Pillows & Throws': 229,
                      'Sports & Outdoors / Exercise & Fitness / Strength Training': 328,
                      'For the Home / Kitchen & Dining / Cutlery': 115,
                      'Entertainment / Books / Cookbooks, Food & Wine': 238,
                      'Health & Beauty / Sexual Wellness / Intimate Apparel & Hosiery': 125,
                      'For the Home / Kitchen & Dining / Table Linens & Placemats': 33,
                      "Men's Fashion / Shoes / Athletic": 255, 'Electronics / Video Games / Games': 183,
                      'Sports & Outdoors / Fan Shop / Premier League': 154,
                      'Entertainment / Magazines / Literature & Writing': 450,
                      "Men's Fashion / Accessories / Ties & Bow Ties": 308, 'For the Home / Home Decor / ': 210,
                      "Men's Fashion / Shoes / Boots": 357, 'For the Home / Luggage / Travel Accessories': 284,
                      'Health & Beauty / Bath & Body / Body Scrubs & Exfoliants': 285,
                      'Health & Beauty / Sexual Wellness / Sexual Supplements': 366,
                      'Auto & Home Improvement / Automotive / Car Safety & Security': 108,
                      'Health & Beauty / Personal Care / Feminine Care': 258,
                      'Health & Beauty / Health Care / Compression': 249, 'Entertainment / Music / R&B': 466,
                      'Sports & Outdoors / Outdoors / Cycling': 301, "Men's Fashion / Clothing / Jeans": 117,
                      'Sports & Outdoors / Golf / Golf Clothing': 350,
                      'Health & Beauty / Vitamins & Supplements / Supplements': 332,
                      'For the Home / Bath / Bath Towels': 165, 'Electronics / Software / Operating Systems': 218,
                      'Entertainment / Movies & TV / Comedy': 69,
                      'Jewelry & Watches / Fine Metal Jewelry / Collections & Sets': 282,
                      'Entertainment / Movies & TV / Health & Fitness': 13,
                      "Women's Fashion / Plus Size Clothing / Tops & Tees": 373,
                      'Electronics / Car Electronics & GPS / Radar Detectors': 400,
                      'For the Home / Bedding / Bed Pillows': 235, 'Health & Beauty / Skin Care / Moisturize': 91,
                      'Sports & Outdoors / Outdoors / Boats & Water Sports': 222,
                      'Grocery, Household & Pets / Alcohol / Wine': 412,
                      'Grocery, Household & Pets / Food / Pantry Items': 230,
                      'Sports & Outdoors / Fan Shop / Olympics': 423,
                      'Health & Beauty / Sexual Wellness / Adult Games': 295,
                      "Women's Fashion / Intimates / Bras": 179, "Men's Fashion / Clothing / Polos": 252,
                      'Electronics / Office & School Supplies / Writing': 15,
                      'For the Home / Storage & Organization / Outdoor': 454,
                      'Auto & Home Improvement / Home Improvement / Plumbing': 365,
                      'Entertainment / Music / Classical': 502, "Women's Fashion / Shoes / Oxfords": 399,
                      'Electronics / Television & Home Theater / Set Top Boxes': 503,
                      'Electronics / Software / Production and Editing': 497,
                      "Women's Fashion / Plus Size Clothing / Dresses": 234,
                      'Electronics / Computers & Tablets / Tablet Accessories': 3,
                      'Electronics / Cell Phones & Accessories / Backup Batteries': 274,
                      'Electronics / Portable Audio / iPod & MP3 Players': 253,
                      'Grocery, Household & Pets / Food / Baby Foods': 73,
                      'Electronics / Musical Instruments / Brass & Woodwinds': 485,
                      'Auto & Home Improvement / Home Improvement / Batteries': 189,
                      "Women's Fashion / Shoes / Pumps & Heels": 60,
                      'Electronics / Musical Instruments / Keyboards & MIDI': 465,
                      'Baby, Kids & Toys / Baby Care / Diapering': 421, 'Electronics / Software / Tax': 481,
                      'Jewelry & Watches / Jewelry Accessories & Storage / Cleaners & Accessories': 463,
                      "Women's Fashion / Intimates / Shapewear": 54, 'Sports & Outdoors / Fan Shop / NHL': 309,
                      'Health & Beauty / Hair Care / Styling Products': 4,
                      "Women's Fashion / Accessories / Hats": 224, 'For the Home / Luggage / Backpacks': 433,
                      'Electronics / Television & Home Theater / Home Audio': 145,
                      "Men's Fashion / Accessories / Sunglasses & Eyewear": 18,
                      'For the Home / Home Decor / Window Treatments': 197,
                      'Sports & Outdoors / Cycling / Parts & Accessories': 403,
                      "Men's Fashion / Shoes / Loafers & Slip-Ons": 134,
                      'For the Home / Kitchen & Dining / Cookware': 174,
                      'Grocery, Household & Pets / Candy & Sweets / Fruity, Gummy & Taffy': 317,
                      "Women's Fashion / Clothing / Tops & Tees": 122,
                      'Sports & Outdoors / Outdoors / Fishing & Marine': 398,
                      'Health & Beauty / Health Care / Medicine Cabinet': 383,
                      "Men's Fashion / Shoes / Boat Shoes": 411,
                      'Health & Beauty / Massage & Relaxation / Foot & Leg Massagers': 270,
                      'Baby, Kids & Toys / Girls Fashion / Jewelry & Watches': 374,
                      'Electronics / Camera, Video & Surveillance / Digital SLRs': 426,
                      'Grocery, Household & Pets / Food / Gourmet Gifts': 314,
                      'Health & Beauty / Cosmetics / Mirrors & Tools': 245}

category_dic2={}
for key,value in category_dic.iteritems():
    category_dic2[value]=key
#print (category_dic2)
lines = open("160819_name_4.txt", "r").read().split('\n')#.decode('utf-8').split('\n')
count=0
count1=0
count0=0
ccq=0
for line in lines:
    sents=line.split('||')
    if (len(sents) <= 2):
            continue
    sent=sents[1]

    #tex = []
    sent = sent.lower()#.encode('ascii', 'ignore')
    sent = "".join(l for l in sent if l not in string.punctuation)
    #sent = sent.translate(None, string.punctuation)
    #sent= str(filter(str.isalpha, sent))
    sent=sent.translate(None, '0123456789')
    sent=sent.replace('’s','')
    sent=str(sent)
    #tex = " ".join([wnl.lemmatize(i) for i in sent.split()])
    #print(sent)
    #sent = ' '.join(wnl.lemmatize(str(word)) for word in sent.split() if wnl.lemmatize(str(word)) not in StopWords)
    sent2=''
    c=0
    for word in sent.split():
        try:
            if wnl.lemmatize(str(word)) not in StopWords:
                sent2=sent2+' '+wnl.lemmatize(str(word))
        except:
            c=c+1

    if c!=0:
       ccq=ccq+1
        #print ("c is",c)
    sent=sent2

    try:
        ww = sents[2].split('>')

        sss2 = str(ww[0]) + str('/') +str(ww[1])+ str('/ ') + str(ww[2]).strip()

        #print(sss2)
        #sss2 = (str(ww[0]) + str('/') + str(ww[1])).strip()
    except:
        #print("count1:",count1)
        print("bbbbbbbbbbb", count1)
        count1=count1+1
         #print (sss2)

    try:
        category_dic.get(sss2)
        #a=doc_dic[category_dic.get(sss2)]
        if doc_dic[category_dic.get(sss2)]==[]:
            doc_dic[category_dic.get(sss2)]=sent
        else:
            doc_dic[category_dic.get(sss2)] = str(doc_dic[category_dic.get(sss2)]) + ' ' + sent
    except:
        #print(sss2)
        print("aaaaaaaa",count)
        count=count+1
    #print("count0:",count0)
    count0=count0+1
    #doc_dic[category_dic.get(sss2)]=str(a)+' '+sent
#print(count)
print ("ccq is.....:",ccq)
#print (doc_dic[36])
for i in range(len(doc_dic)):
    if doc_dic[i]==[]:
        continue
    vocab=vocab+doc_dic[i].split()


vocab = list(set(vocab))
print(len(vocab))


def shape(A):
    num_rows=len(A)
    num_cols=len(A[0]) if A else 0

    return num_rows,num_cols



def make_matrix(num_rows,num_cols,entry_fn):
    return [[entry_fn(i,j) for j in range(num_cols)]for i in range(num_rows)]


def is_diagonal(i,j):

    if doc_dic[i] == []:
        return 0
    return doc_dic[i].split().count(vocab[j])

dtm=make_matrix(len(doc_dic),len(vocab),is_diagonal)
print("dtm shape")
print (shape(dtm))
doccount=shape(dtm)[0]
vocabcount=shape(dtm)[1]
dtmcount=np.zeros((doccount,1))
for i in range(doccount):
    for j in range(vocabcount):
        if dtm[i][j]>0:
            dtmcount[i][0]+=1
for t in range(doccount):
    print(category_dic2[t],dtmcount[t][0])

global n
n=[]


d=[]
for b in range(doccount):
    sum=0
    sum2=0
    for a in dtm[b]:
        if a>0:
            sum=sum+a
    n.append(sum)

for a in range(vocabcount):
    sum = 0
    for i in range(doccount):
        if dtm[i][a]>0:
            sum=sum+1
   # print(sum)
    d.append(np.log10(float(doccount+1)/float(sum)))


def tf_idf(i):
    tfidf={}
    if n[i]==0:
        return
    for j in range(vocabcount):
        tfidf[vocab[j]] = float(float(dtm[i][j]) / float(n[i]) * d[j])

    new=sorted(tfidf.iteritems(), key=itemgetter(1), reverse=True)
    abc=[]
    #abc1=[]
    abc2=[]
    count=0
    #print(category_dic2.get(i)+'=')
    for j in new:
        #print(j)
        if j[1] > 0:
            if count >= 50:
            # abc1.append(j[0])
                abc2.append(j)
                count = count + 1
            elif count<50:
                abc.append(j[0])
                # abc1.append(j[0])
                abc2.append(j)
                count = count + 1
        if j[1]<0:
            break

    tro.write(str(count))
    tro.write('||')
    tro.write(category_dic2.get(i))
    tro.write('||')
    """t.write('=')
    t.write('[')
    for a in abc1:
        t.write("'")
        t.write(str(a))
        t.write("',")
    t.write(']')
    t.write('\n')"""
   # t.write('')
    for a in abc2:
        tro.write(str(a[0]))
        tro.write(">")
        tro.write(str(a[1]))
        tro.write('#')

    tro.write('\n')
    g.write(category_dic2.get(i))
    g.write('=')
    for a in abc:
        g.write(str(a))
        g.write(",")
    g.write('\n')
    #print (abc)


for i in range(len(doc_dic)):


    tf_idf(i)
