"""
Boolean Expressions Task - Configuration and Data

Contains facts data, prompt templates, and parameter derivation.
"""

import random

# ============================================================================
# Common Sense Facts (Fixed lists)
# ============================================================================

COMMON_SENSE_TRUE_EN = [
    "The Earth revolves around the Sun.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The human body has 206 bones.",
    "Oxygen is necessary for human survival.",
    "Sound cannot travel through a vacuum.",
    "Diamonds are made of carbon.",
    "Antarctica is the coldest continent.",
    "A day on Earth is 24 hours.",
    "Humans have five fingers on each hand.",
    "The Pacific Ocean is the largest ocean.",
    "Light travels faster than sound.",
    "The Moon orbits the Earth.",
    "Ice is less dense than liquid water.",
    "Plants produce oxygen through photosynthesis.",
    "The speed of light is approximately 300,000 km/s.",
]

COMMON_SENSE_FALSE_EN = [
    "The Sun revolves around the Earth.",
    "Humans have three lungs.",
    "Spiders are insects.",
    "Sharks are mammals.",
    "Penguins can fly.",
    "The Great Wall of China was built in one year.",
    "Electricity flows faster than light.",
    "Sound travels faster in a vacuum.",
    "The human body has 150 bones.",
    "Mars is the closest planet to the Sun.",
    "Whales are fish.",
    "Bats are blind.",
    "Lightning never strikes the same place twice.",
    "The Moon produces its own light.",
    "Humans only use 10% of their brain.",
]

COMMON_SENSE_TRUE_CN = [
    "地球绕太阳运转。",
    "水在海平面上的沸点是100摄氏度。",
    "人体有206块骨头。",
    "氧气对人类生存是必需的。",
    "声音无法在真空中传播。",
    "钻石由碳元素组成。",
    "南极洲是最冷的大陆。",
    "地球上一天是24小时。",
    "人类每只手有五个手指。",
    "太平洋是最大的海洋。",
    "光速比声速快。",
    "月球绕地球运转。",
    "冰的密度比液态水小。",
    "植物通过光合作用产生氧气。",
    "光速约为每秒30万公里。",
]

COMMON_SENSE_FALSE_CN = [
    "太阳绕地球运转。",
    "人类有三个肺。",
    "蜘蛛是昆虫。",
    "鲨鱼是哺乳动物。",
    "企鹅能飞翔。",
    "中国长城是在一年内建成的。",
    "电流比光速更快。",
    "声音在真空中传播更快。",
    "人体有150块骨头。",
    "火星是距离太阳最近的行星。",
    "鲸鱼是鱼类。",
    "蝙蝠是瞎的。",
    "闪电不会两次击中同一个地方。",
    "月球自己发光。",
    "人类只使用了大脑的10%。",
]

# ============================================================================
# Geography Data (Template-based generation)
# ============================================================================

# Country -> Capital mapping
CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "China": "Beijing",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Russia": "Moscow",
    "Brazil": "Brasilia",
    "India": "New Delhi",
    "Australia": "Canberra",
    "Canada": "Ottawa",
    "Mexico": "Mexico City",
    "Egypt": "Cairo",
    "South Korea": "Seoul",
    "Argentina": "Buenos Aires",
    "Turkey": "Ankara",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Poland": "Warsaw",
    "Netherlands": "Amsterdam",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Denmark": "Copenhagen",
    "Finland": "Helsinki",
    "Greece": "Athens",
    "Portugal": "Lisbon",
    "Austria": "Vienna",
    "Switzerland": "Bern",
    "Belgium": "Brussels",
    "Ireland": "Dublin",
    "New Zealand": "Wellington",
    "Singapore": "Singapore",
    "Malaysia": "Kuala Lumpur",
    "Indonesia": "Jakarta",
    "Philippines": "Manila",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "Morocco": "Rabat",
    "Chile": "Santiago",
    "Colombia": "Bogota",
    "Peru": "Lima",
    "Venezuela": "Caracas",
    "Cuba": "Havana",
    "Czech Republic": "Prague",
    "Hungary": "Budapest",
    "Romania": "Bucharest",
    "Ukraine": "Kyiv",
    "Saudi Arabia": "Riyadh",
}

CAPITALS_CN = {
    "法国": "巴黎",
    "日本": "东京",
    "中国": "北京",
    "德国": "柏林",
    "意大利": "罗马",
    "西班牙": "马德里",
    "俄罗斯": "莫斯科",
    "巴西": "巴西利亚",
    "印度": "新德里",
    "澳大利亚": "堪培拉",
    "加拿大": "渥太华",
    "墨西哥": "墨西哥城",
    "埃及": "开罗",
    "韩国": "首尔",
    "阿根廷": "布宜诺斯艾利斯",
    "土耳其": "安卡拉",
    "泰国": "曼谷",
    "越南": "河内",
    "波兰": "华沙",
    "荷兰": "阿姆斯特丹",
    "瑞典": "斯德哥尔摩",
    "挪威": "奥斯陆",
    "丹麦": "哥本哈根",
    "芬兰": "赫尔辛基",
    "希腊": "雅典",
    "葡萄牙": "里斯本",
    "奥地利": "维也纳",
    "瑞士": "伯尔尼",
    "比利时": "布鲁塞尔",
    "爱尔兰": "都柏林",
    "新西兰": "惠灵顿",
    "新加坡": "新加坡",
    "马来西亚": "吉隆坡",
    "印度尼西亚": "雅加达",
    "菲律宾": "马尼拉",
    "尼日利亚": "阿布贾",
    "肯尼亚": "内罗毕",
    "摩洛哥": "拉巴特",
    "智利": "圣地亚哥",
    "哥伦比亚": "波哥大",
    "秘鲁": "利马",
    "委内瑞拉": "加拉加斯",
    "古巴": "哈瓦那",
    "捷克": "布拉格",
    "匈牙利": "布达佩斯",
    "罗马尼亚": "布加勒斯特",
    "乌克兰": "基辅",
    "沙特阿拉伯": "利雅得",
}

# ============================================================================
# Science Data (Element symbols)
# ============================================================================

ELEMENTS = {
    "Gold": "Au",
    "Silver": "Ag",
    "Iron": "Fe",
    "Copper": "Cu",
    "Zinc": "Zn",
    "Lead": "Pb",
    "Tin": "Sn",
    "Mercury": "Hg",
    "Platinum": "Pt",
    "Sodium": "Na",
    "Potassium": "K",
    "Calcium": "Ca",
    "Magnesium": "Mg",
    "Aluminum": "Al",
    "Carbon": "C",
    "Nitrogen": "N",
    "Oxygen": "O",
    "Hydrogen": "H",
    "Helium": "He",
    "Neon": "Ne",
    "Chlorine": "Cl",
    "Sulfur": "S",
    "Phosphorus": "P",
    "Silicon": "Si",
    "Nickel": "Ni",
    "Cobalt": "Co",
    "Chromium": "Cr",
    "Manganese": "Mn",
    "Titanium": "Ti",
    "Uranium": "U",
}

ELEMENTS_CN = {
    "金": "Au",
    "银": "Ag",
    "铁": "Fe",
    "铜": "Cu",
    "锌": "Zn",
    "铅": "Pb",
    "锡": "Sn",
    "汞": "Hg",
    "铂": "Pt",
    "钠": "Na",
    "钾": "K",
    "钙": "Ca",
    "镁": "Mg",
    "铝": "Al",
    "碳": "C",
    "氮": "N",
    "氧": "O",
    "氢": "H",
    "氦": "He",
    "氖": "Ne",
    "氯": "Cl",
    "硫": "S",
    "磷": "P",
    "硅": "Si",
    "镍": "Ni",
    "钴": "Co",
    "铬": "Cr",
    "锰": "Mn",
    "钛": "Ti",
    "铀": "U",
}

# ============================================================================
# Time Data
# ============================================================================

DAYS_IN_MONTH = {
    "January": 31,
    "February": 28,  # Non-leap year
    "March": 31,
    "April": 30,
    "May": 31,
    "June": 30,
    "July": 31,
    "August": 31,
    "September": 30,
    "October": 31,
    "November": 30,
    "December": 31,
}

DAYS_IN_MONTH_CN = {
    "一月": 31,
    "二月": 28,
    "三月": 31,
    "四月": 30,
    "五月": 31,
    "六月": 30,
    "七月": 31,
    "八月": 31,
    "九月": 30,
    "十月": 31,
    "十一月": 30,
    "十二月": 31,
}

# Leap years from 1900-2100
LEAP_YEARS = {year for year in range(1904, 2100, 4) if (year % 100 != 0 or year % 400 == 0)}

# ============================================================================
# Math Data (Primes for fact generation)
# ============================================================================

PRIMES_UNDER_100 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}

# Perfect squares
PERFECT_SQUARES = {i * i for i in range(1, 20)}  # 1, 4, 9, 16, ..., 361

# Fibonacci numbers under 1000
FIBONACCI_NUMBERS = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987}

# ============================================================================
# Astronomy Data
# ============================================================================

# Planet order from Sun (1-indexed)
PLANET_ORDER = {
    "Mercury": 1,
    "Venus": 2,
    "Earth": 3,
    "Mars": 4,
    "Jupiter": 5,
    "Saturn": 6,
    "Uranus": 7,
    "Neptune": 8,
}

PLANET_ORDER_CN = {
    "水星": 1,
    "金星": 2,
    "地球": 3,
    "火星": 4,
    "木星": 5,
    "土星": 6,
    "天王星": 7,
    "海王星": 8,
}

# ============================================================================
# Biology Data
# ============================================================================

# Animal legs count
ANIMAL_LEGS = {
    "spider": 8,
    "ant": 6,
    "dog": 4,
    "cat": 4,
    "bird": 2,
    "human": 2,
    "snake": 0,
    "fish": 0,
    "octopus": 8,
    "crab": 10,
    "butterfly": 6,
    "horse": 4,
    "elephant": 4,
    "kangaroo": 2,
    "frog": 4,
    "chicken": 2,
    "cow": 4,
    "pig": 4,
    "duck": 2,
    "lobster": 10,
}

ANIMAL_LEGS_CN = {
    "蜘蛛": 8,
    "蚂蚁": 6,
    "狗": 4,
    "猫": 4,
    "鸟": 2,
    "人类": 2,
    "蛇": 0,
    "鱼": 0,
    "章鱼": 8,
    "螃蟹": 10,
    "蝴蝶": 6,
    "马": 4,
    "大象": 4,
    "袋鼠": 2,
    "青蛙": 4,
    "鸡": 2,
    "牛": 4,
    "猪": 4,
    "鸭子": 2,
    "龙虾": 10,
}

# Animal classification
MAMMALS = {"dog", "cat", "whale", "dolphin", "bat", "elephant", "lion", "tiger", "bear", "human", "horse", "cow", "pig", "sheep", "rabbit", "mouse", "kangaroo", "koala"}
MAMMALS_CN = {"狗", "猫", "鲸鱼", "海豚", "蝙蝠", "大象", "狮子", "老虎", "熊", "人类", "马", "牛", "猪", "羊", "兔子", "老鼠", "袋鼠", "考拉"}

BIRDS = {"eagle", "sparrow", "penguin", "ostrich", "owl", "parrot", "crow", "duck", "chicken", "peacock", "flamingo", "swan"}
BIRDS_CN = {"鹰", "麻雀", "企鹅", "鸵鸟", "猫头鹰", "鹦鹉", "乌鸦", "鸭子", "鸡", "孔雀", "火烈鸟", "天鹅"}

REPTILES = {"snake", "lizard", "crocodile", "turtle", "gecko", "iguana", "chameleon", "alligator"}
REPTILES_CN = {"蛇", "蜥蜴", "鳄鱼", "乌龟", "壁虎", "鬣蜥", "变色龙", "短吻鳄"}

FISH = {"salmon", "tuna", "shark", "goldfish", "cod", "herring", "carp", "trout"}
FISH_CN = {"三文鱼", "金枪鱼", "鲨鱼", "金鱼", "鳕鱼", "鲱鱼", "鲤鱼", "鳟鱼"}

INSECTS = {"ant", "bee", "butterfly", "mosquito", "fly", "beetle", "grasshopper", "dragonfly", "cockroach", "ladybug"}
INSECTS_CN = {"蚂蚁", "蜜蜂", "蝴蝶", "蚊子", "苍蝇", "甲虫", "蚱蜢", "蜻蜓", "蟑螂", "瓢虫"}

# ============================================================================
# History Data
# ============================================================================

HISTORICAL_EVENTS = {
    "World War I ended": 1918,
    "World War II ended": 1945,
    "The first moon landing": 1969,
    "The Berlin Wall fell": 1989,
    "The French Revolution began": 1789,
    "Christopher Columbus reached America": 1492,
    "The United States declared independence": 1776,
    "The Soviet Union dissolved": 1991,
    "World War I began": 1914,
    "World War II began": 1939,
    "The first airplane flight by the Wright Brothers": 1903,
    "The Titanic sank": 1912,
    "The first iPhone was released": 2007,
    "The first Olympic Games of modern era": 1896,
    "The Great Fire of London": 1666,
}

HISTORICAL_EVENTS_CN = {
    "第一次世界大战结束": 1918,
    "第二次世界大战结束": 1945,
    "人类首次登月": 1969,
    "柏林墙倒塌": 1989,
    "法国大革命开始": 1789,
    "哥伦布到达美洲": 1492,
    "美国宣布独立": 1776,
    "苏联解体": 1991,
    "第一次世界大战开始": 1914,
    "第二次世界大战开始": 1939,
    "莱特兄弟首次飞行": 1903,
    "泰坦尼克号沉没": 1912,
    "第一部iPhone发布": 2007,
    "现代奥运会首届举办": 1896,
    "伦敦大火": 1666,
}

# ============================================================================
# Physics Data
# ============================================================================

# States of matter at room temperature (20°C)
STATES_OF_MATTER = {
    "water": "liquid",
    "ice": "solid",
    "steam": "gas",
    "iron": "solid",
    "gold": "solid",
    "mercury": "liquid",
    "oxygen": "gas",
    "nitrogen": "gas",
    "carbon dioxide": "gas",
    "wood": "solid",
    "oil": "liquid",
    "alcohol": "liquid",
    "helium": "gas",
    "copper": "solid",
    "silver": "solid",
    "diamond": "solid",
    "air": "gas",
}

STATES_OF_MATTER_CN = {
    "水": "液体",
    "冰": "固体",
    "水蒸气": "气体",
    "铁": "固体",
    "金": "固体",
    "汞": "液体",
    "氧气": "气体",
    "氮气": "气体",
    "二氧化碳": "气体",
    "木头": "固体",
    "油": "液体",
    "酒精": "液体",
    "氦气": "气体",
    "铜": "固体",
    "银": "固体",
    "钻石": "固体",
    "空气": "气体",
}

STATE_NAMES_EN = ["solid", "liquid", "gas"]
STATE_NAMES_CN = ["固体", "液体", "气体"]

# ============================================================================
# Extended Geography Data
# ============================================================================

# Countries and their continents
COUNTRY_CONTINENT = {
    "China": "Asia",
    "Japan": "Asia",
    "India": "Asia",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Brazil": "South America",
    "Argentina": "South America",
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Australia": "Oceania",
    "Egypt": "Africa",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Russia": "Europe",
    "United Kingdom": "Europe",
    "Spain": "Europe",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "South Korea": "Asia",
    "Indonesia": "Asia",
    "Saudi Arabia": "Asia",
    "Kenya": "Africa",
    "Morocco": "Africa",
    "Chile": "South America",
    "Colombia": "South America",
    "Peru": "South America",
    "New Zealand": "Oceania",
}

COUNTRY_CONTINENT_CN = {
    "中国": "亚洲",
    "日本": "亚洲",
    "印度": "亚洲",
    "法国": "欧洲",
    "德国": "欧洲",
    "意大利": "欧洲",
    "巴西": "南美洲",
    "阿根廷": "南美洲",
    "美国": "北美洲",
    "加拿大": "北美洲",
    "墨西哥": "北美洲",
    "澳大利亚": "大洋洲",
    "埃及": "非洲",
    "南非": "非洲",
    "尼日利亚": "非洲",
    "俄罗斯": "欧洲",
    "英国": "欧洲",
    "西班牙": "欧洲",
    "泰国": "亚洲",
    "越南": "亚洲",
    "韩国": "亚洲",
    "印度尼西亚": "亚洲",
    "沙特阿拉伯": "亚洲",
    "肯尼亚": "非洲",
    "摩洛哥": "非洲",
    "智利": "南美洲",
    "哥伦比亚": "南美洲",
    "秘鲁": "南美洲",
    "新西兰": "大洋洲",
}

CONTINENTS_EN = ["Asia", "Europe", "Africa", "North America", "South America", "Oceania", "Antarctica"]
CONTINENTS_CN = ["亚洲", "欧洲", "非洲", "北美洲", "南美洲", "大洋洲", "南极洲"]

# ============================================================================
# Currency Data
# ============================================================================

COUNTRY_CURRENCY = {
    "United States": "Dollar",
    "Japan": "Yen",
    "United Kingdom": "Pound",
    "European Union": "Euro",
    "China": "Yuan",
    "India": "Rupee",
    "Russia": "Ruble",
    "Brazil": "Real",
    "Australia": "Dollar",
    "Canada": "Dollar",
    "Switzerland": "Franc",
    "South Korea": "Won",
    "Mexico": "Peso",
    "Singapore": "Dollar",
    "Thailand": "Baht",
    "Malaysia": "Ringgit",
    "Indonesia": "Rupiah",
    "Philippines": "Peso",
    "Vietnam": "Dong",
    "South Africa": "Rand",
    "Turkey": "Lira",
    "Saudi Arabia": "Riyal",
    "Egypt": "Pound",
    "Nigeria": "Naira",
    "Argentina": "Peso",
    "Chile": "Peso",
    "Colombia": "Peso",
    "Peru": "Sol",
    "Sweden": "Krona",
    "Norway": "Krone",
    "Denmark": "Krone",
    "Poland": "Zloty",
    "Czech Republic": "Koruna",
    "Hungary": "Forint",
    "Israel": "Shekel",
    "New Zealand": "Dollar",
}

COUNTRY_CURRENCY_CN = {
    "美国": "美元",
    "日本": "日元",
    "英国": "英镑",
    "欧盟": "欧元",
    "中国": "人民币",
    "印度": "卢比",
    "俄罗斯": "卢布",
    "巴西": "雷亚尔",
    "澳大利亚": "澳元",
    "加拿大": "加元",
    "瑞士": "瑞士法郎",
    "韩国": "韩元",
    "墨西哥": "比索",
    "新加坡": "新加坡元",
    "泰国": "泰铢",
    "马来西亚": "林吉特",
    "印度尼西亚": "印尼盾",
    "菲律宾": "比索",
    "越南": "越南盾",
    "南非": "兰特",
    "土耳其": "里拉",
    "沙特阿拉伯": "里亚尔",
    "埃及": "埃及镑",
    "尼日利亚": "奈拉",
    "阿根廷": "比索",
    "智利": "比索",
    "哥伦比亚": "比索",
    "秘鲁": "索尔",
    "瑞典": "克朗",
    "挪威": "克朗",
    "丹麦": "克朗",
    "波兰": "兹罗提",
    "捷克": "克朗",
    "匈牙利": "福林",
    "以色列": "谢克尔",
    "新西兰": "新西兰元",
}

# ============================================================================
# Sports/Olympics Data
# ============================================================================

OLYMPICS_HOST_CITIES = {
    1896: "Athens",
    1900: "Paris",
    1904: "St. Louis",
    1908: "London",
    1912: "Stockholm",
    1920: "Antwerp",
    1924: "Paris",
    1928: "Amsterdam",
    1932: "Los Angeles",
    1936: "Berlin",
    1948: "London",
    1952: "Helsinki",
    1956: "Melbourne",
    1960: "Rome",
    1964: "Tokyo",
    1968: "Mexico City",
    1972: "Munich",
    1976: "Montreal",
    1980: "Moscow",
    1984: "Los Angeles",
    1988: "Seoul",
    1992: "Barcelona",
    1996: "Atlanta",
    2000: "Sydney",
    2004: "Athens",
    2008: "Beijing",
    2012: "London",
    2016: "Rio de Janeiro",
    2020: "Tokyo",
    2024: "Paris",
}

OLYMPICS_HOST_CITIES_CN = {
    1896: "雅典",
    1900: "巴黎",
    1904: "圣路易斯",
    1908: "伦敦",
    1912: "斯德哥尔摩",
    1920: "安特卫普",
    1924: "巴黎",
    1928: "阿姆斯特丹",
    1932: "洛杉矶",
    1936: "柏林",
    1948: "伦敦",
    1952: "赫尔辛基",
    1956: "墨尔本",
    1960: "罗马",
    1964: "东京",
    1968: "墨西哥城",
    1972: "慕尼黑",
    1976: "蒙特利尔",
    1980: "莫斯科",
    1984: "洛杉矶",
    1988: "首尔",
    1992: "巴塞罗那",
    1996: "亚特兰大",
    2000: "悉尼",
    2004: "雅典",
    2008: "北京",
    2012: "伦敦",
    2016: "里约热内卢",
    2020: "东京",
    2024: "巴黎",
}

# ============================================================================
# Prompt Templates
# ============================================================================

PROMPTS_EN = [
    "Evaluate the following boolean expressions and select the ones that are true:\n\n{options}\n\nWhich options evaluate to True? List all true option labels separated by commas. Put your final answer in \\boxed{{}}.",
    "Consider the following logical expressions:\n\n{options}\n\nDetermine which expressions are true. List all true option labels separated by commas in \\boxed{{}}.",
    "Analyze these boolean expressions:\n\n{options}\n\nWhich of these evaluate to True? Provide all true option labels separated by commas in \\boxed{{}}.",
    "Below are several boolean expressions:\n\n{options}\n\nIdentify all expressions that are true. Answer with the option labels separated by commas in \\boxed{{}}.",
    "Evaluate each expression below:\n\n{options}\n\nList all options that evaluate to True, separated by commas, in \\boxed{{}}.",
    "Study the following logical statements:\n\n{options}\n\nWhich statements are true? Put all true option labels in \\boxed{{}}, separated by commas.",
    "Here are some boolean expressions to evaluate:\n\n{options}\n\nFind all expressions that are True. Answer in \\boxed{{}} with labels separated by commas.",
    "Examine these expressions:\n\n{options}\n\nWhich ones are true? List the labels of all true options in \\boxed{{}}, separated by commas.",
    "Given the following boolean expressions:\n\n{options}\n\nSelect all that evaluate to True. Put your answer in \\boxed{{}} as comma-separated labels.",
    "Review these logical expressions:\n\n{options}\n\nWhich expressions are true? Answer with comma-separated labels in \\boxed{{}}.",
]

PROMPTS_CN = [
    "请评估以下布尔表达式，选择其中为真的选项：\n\n{options}\n\n哪些选项的值为真？请列出所有为真的选项标识，用逗号分隔，填入 \\boxed{{}} 中。",
    "分析以下逻辑表达式：\n\n{options}\n\n判断哪些表达式为真。在 \\boxed{{}} 中列出所有为真的选项，用逗号分隔。",
    "考虑以下布尔表达式：\n\n{options}\n\n哪些表达式的值为 True？请在 \\boxed{{}} 中给出答案，用逗号分隔选项标识。",
    "下面是几个布尔表达式：\n\n{options}\n\n找出所有为真的表达式。在 \\boxed{{}} 中用逗号分隔列出选项标识。",
    "评估以下每个表达式：\n\n{options}\n\n列出所有值为 True 的选项，在 \\boxed{{}} 中用逗号分隔。",
    "研究以下逻辑语句：\n\n{options}\n\n哪些语句为真？在 \\boxed{{}} 中填入所有为真的选项标识，用逗号分隔。",
    "这里有一些需要评估的布尔表达式：\n\n{options}\n\n找出所有为 True 的表达式。在 \\boxed{{}} 中给出答案，用逗号分隔。",
    "检验这些表达式：\n\n{options}\n\n哪些是真的？在 \\boxed{{}} 中列出所有为真选项的标识，用逗号分隔。",
    "给定以下布尔表达式：\n\n{options}\n\n选出所有值为 True 的选项。在 \\boxed{{}} 中以逗号分隔的形式给出答案。",
    "审视以下逻辑表达式：\n\n{options}\n\n哪些表达式为真？在 \\boxed{{}} 中用逗号分隔给出答案。",
]

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# ============================================================================
# Parameter Derivation
# ============================================================================


def derive_params_from_seed(seed: int) -> dict:
    """
    Derive all generation parameters from seed.

    Args:
        seed: Random seed

    Returns:
        dict with all parameters
    """
    rng = random.Random(seed)

    return {
        "depth": rng.randint(2, 6),
        "num_options": rng.randint(3, 8),
        "language": rng.choice(["en", "cn"]),
        "prompt_idx": rng.randint(0, 9),
        # Weights for different fact types
        "fact_type_weights": [0.15, 0.25, 0.25, 0.15, 0.20],  # common, geo, math, time, science
    }


def get_prompts(language: str) -> list:
    """Get prompt templates for given language."""
    return PROMPTS_CN if language == "cn" else PROMPTS_EN


def get_common_sense_facts(language: str) -> tuple:
    """Get common sense facts for given language."""
    if language == "cn":
        return COMMON_SENSE_TRUE_CN, COMMON_SENSE_FALSE_CN
    return COMMON_SENSE_TRUE_EN, COMMON_SENSE_FALSE_EN


def get_capitals(language: str) -> dict:
    """Get capitals data for given language."""
    return CAPITALS_CN if language == "cn" else CAPITALS


def get_elements(language: str) -> dict:
    """Get elements data for given language."""
    return ELEMENTS_CN if language == "cn" else ELEMENTS


def get_days_in_month(language: str) -> dict:
    """Get days in month data for given language."""
    return DAYS_IN_MONTH_CN if language == "cn" else DAYS_IN_MONTH


def get_planet_order(language: str) -> dict:
    """Get planet order data for given language."""
    return PLANET_ORDER_CN if language == "cn" else PLANET_ORDER


def get_animal_legs(language: str) -> dict:
    """Get animal legs data for given language."""
    return ANIMAL_LEGS_CN if language == "cn" else ANIMAL_LEGS


def get_animal_classes(language: str) -> dict:
    """Get animal classification data for given language."""
    if language == "cn":
        return {
            "mammals": MAMMALS_CN,
            "birds": BIRDS_CN,
            "reptiles": REPTILES_CN,
            "fish": FISH_CN,
            "insects": INSECTS_CN,
        }
    return {
        "mammals": MAMMALS,
        "birds": BIRDS,
        "reptiles": REPTILES,
        "fish": FISH,
        "insects": INSECTS,
    }


def get_historical_events(language: str) -> dict:
    """Get historical events data for given language."""
    return HISTORICAL_EVENTS_CN if language == "cn" else HISTORICAL_EVENTS


def get_states_of_matter(language: str) -> tuple:
    """Get states of matter data for given language."""
    if language == "cn":
        return STATES_OF_MATTER_CN, STATE_NAMES_CN
    return STATES_OF_MATTER, STATE_NAMES_EN


def get_country_continent(language: str) -> tuple:
    """Get country continent data for given language."""
    if language == "cn":
        return COUNTRY_CONTINENT_CN, CONTINENTS_CN
    return COUNTRY_CONTINENT, CONTINENTS_EN


def get_country_currency(language: str) -> dict:
    """Get country currency data for given language."""
    return COUNTRY_CURRENCY_CN if language == "cn" else COUNTRY_CURRENCY


def get_olympics_data(language: str) -> dict:
    """Get Olympics host cities data for given language."""
    return OLYMPICS_HOST_CITIES_CN if language == "cn" else OLYMPICS_HOST_CITIES
