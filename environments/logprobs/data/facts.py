"""Common-sense Q&A fact pool for the common_sense_combo task.

Each entry: (question_en, question_zh, answer_en, answer_zh)
Open-ended questions — no forced answer format, model can respond freely.
"""

# ---------------------------------------------------------------------------
# World Geography
# ---------------------------------------------------------------------------
GEOGRAPHY = [
    ("What is the longest river in the world?",
     "世界上最长的河流是什么？",
     "The Nile", "尼罗河"),
    ("What is the highest mountain on Earth?",
     "地球上最高的山是哪座？",
     "Mount Everest", "珠穆朗玛峰"),
    ("What is the largest ocean on Earth?",
     "地球上最大的海洋是什么？",
     "The Pacific Ocean", "太平洋"),
    ("What is the largest country by area?",
     "面积最大的国家是哪个？",
     "Russia", "俄罗斯"),
    ("What is the smallest country in the world?",
     "世界上最小的国家是哪个？",
     "Vatican City", "梵蒂冈"),
    ("What is the largest hot desert in the world?",
     "世界上最大的热沙漠是什么？",
     "The Sahara", "撒哈拉沙漠"),
    ("What is the deepest point in the ocean?",
     "海洋的最深处是哪里？",
     "The Mariana Trench", "马里亚纳海沟"),
    ("What is the lowest point on Earth's surface?",
     "地球表面最低点在哪里？",
     "The Dead Sea", "死海"),
    ("Which continent has the largest area?",
     "哪个大洲面积最大？",
     "Asia", "亚洲"),
    ("What country is both a continent and a nation?",
     "哪个国家同时也是一个大洲？",
     "Australia", "澳大利亚"),
]

# ---------------------------------------------------------------------------
# Science & Nature
# ---------------------------------------------------------------------------
SCIENCE = [
    ("At what temperature does water boil at sea level?",
     "水在海平面的沸点是多少？",
     "100 degrees Celsius", "100摄氏度"),
    ("At what temperature does water freeze?",
     "水在多少度结冰？",
     "0 degrees Celsius", "0摄氏度"),
    ("What is the approximate speed of light in a vacuum?",
     "光在真空中的速度大约是多少？",
     "About 300,000 km/s", "约30万公里每秒"),
    ("What element does the chemical symbol Fe represent?",
     "化学符号Fe代表哪种元素？",
     "Iron", "铁"),
    ("What element does the chemical symbol Na represent?",
     "化学符号Na代表哪种元素？",
     "Sodium", "钠"),
    ("What element does the chemical symbol Au represent?",
     "化学符号Au代表哪种元素？",
     "Gold", "金"),
    ("What is the chemical formula for water?",
     "水的化学式是什么？",
     "H2O", "H2O"),
    ("How many elements are in the periodic table?",
     "元素周期表中有多少种元素？",
     "118", "118种"),
    ("What is DNA an abbreviation for?",
     "DNA是什么的缩写？",
     "Deoxyribonucleic acid", "脱氧核糖核酸"),
    ("How long does light from the Sun take to reach Earth?",
     "太阳光到达地球需要多长时间？",
     "About 8 minutes", "约8分钟"),
    ("What type of star is the Sun?",
     "太阳是什么类型的恒星？",
     "A yellow dwarf star", "黄矮星"),
    ("What gas makes up most of Earth's atmosphere?",
     "地球大气层中含量最多的气体是什么？",
     "Nitrogen", "氮气"),
]

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
HISTORY = [
    ("In what year did World War II end?",
     "第二次世界大战哪年结束？",
     "1945", "1945年"),
    ("In what year did World War I begin?",
     "第一次世界大战哪年开始？",
     "1914", "1914年"),
    ("In what year did the Berlin Wall fall?",
     "柏林墙哪年倒塌？",
     "1989", "1989年"),
    ("Who was the first person to walk on the Moon?",
     "谁是第一个在月球上行走的人？",
     "Neil Armstrong", "尼尔·阿姆斯特朗"),
    ("In what year did humans first land on the Moon?",
     "人类首次登月是哪年？",
     "1969", "1969年"),
    ("Who invented the telephone?",
     "电话是谁发明的？",
     "Alexander Graham Bell", "亚历山大·格雷厄姆·贝尔"),
    ("In what year did Christopher Columbus reach the Americas?",
     "哥伦布哪年到达美洲？",
     "1492", "1492年"),
    ("When did the French Revolution begin?",
     "法国大革命始于哪年？",
     "1789", "1789年"),
    ("In what year did the Titanic sink?",
     "泰坦尼克号哪年沉没？",
     "1912", "1912年"),
    ("When was the first artificial satellite, Sputnik, launched?",
     "第一颗人造卫星斯普特尼克哪年发射？",
     "1957", "1957年"),
]

# ---------------------------------------------------------------------------
# Culture & Everyday Knowledge
# ---------------------------------------------------------------------------
CULTURE = [
    ("How many keys does a standard piano have?",
     "标准钢琴有多少个琴键？",
     "88", "88个"),
    ("How many squares are on a standard chess board?",
     "标准国际象棋棋盘有多少个方格？",
     "64", "64个"),
    ("How many players are on a soccer team on the field?",
     "足球场上每队有多少名球员？",
     "11", "11名"),
    ("How many players are on a basketball team on the court?",
     "篮球场上每队有多少名球员？",
     "5", "5名"),
    ("How many cards are in a standard deck (excluding jokers)?",
     "标准扑克牌（不含大小王）有多少张？",
     "52", "52张"),
    ("Who painted the Mona Lisa?",
     "蒙娜丽莎是谁画的？",
     "Leonardo da Vinci", "列奥纳多·达·芬奇"),
    ("Who wrote Romeo and Juliet?",
     "罗密欧与朱丽叶是谁写的？",
     "William Shakespeare", "威廉·莎士比亚"),
    ("Where is the Eiffel Tower located?",
     "埃菲尔铁塔在哪里？",
     "Paris, France", "法国巴黎"),
    ("What country gifted the Statue of Liberty to the United States?",
     "自由女神像是哪个国家赠送给美国的？",
     "France", "法国"),
    ("Who composed the Fifth Symphony?",
     "第五交响曲是谁创作的？",
     "Ludwig van Beethoven", "路德维希·凡·贝多芬"),
    ("Who wrote the Harry Potter series?",
     "哈利·波特系列是谁写的？",
     "J.K. Rowling", "J.K.罗琳"),
    ("What country does sushi originate from?",
     "寿司起源于哪个国家？",
     "Japan", "日本"),
]

# ---------------------------------------------------------------------------
# Human Body & Biology
# ---------------------------------------------------------------------------
BIOLOGY = [
    ("How many bones does an adult human body have?",
     "成年人体有多少块骨头？",
     "206", "206块"),
    ("How many chambers does the human heart have?",
     "人类心脏有几个腔室？",
     "4", "4个"),
    ("How many adult teeth do humans typically have?",
     "人类通常有多少颗恒牙？",
     "32", "32颗"),
    ("What is the largest internal organ in the human body?",
     "人体最大的内部器官是什么？",
     "The liver", "肝脏"),
    ("What percentage of the body's energy does the human brain use?",
     "人脑消耗身体多少比例的能量？",
     "About 20%", "约20%"),
    ("What do red blood cells carry?",
     "红细胞携带什么？",
     "Oxygen", "氧气"),
    ("Approximately what percentage of DNA do humans share with chimpanzees?",
     "人类与黑猩猩大约有多少比例的DNA相同？",
     "About 98%", "约98%"),
]

# ---------------------------------------------------------------------------
# Prompt templates (open-ended, no forced answer format)
# ---------------------------------------------------------------------------

TEMPLATES_EN = [
    "Please answer the following questions:\n{items}",
    "Answer each question below as accurately as you can:\n{items}",
    "Briefly answer the following:\n{items}",
    "Respond to each of these questions:\n{items}",
    "Give a short answer to each question:\n{items}",
]

TEMPLATES_ZH = [
    "请回答以下问题：\n{items}",
    "请尽量准确地回答下列问题：\n{items}",
    "请简要回答以下每个问题：\n{items}",
    "请依次作答：\n{items}",
    "请对以下问题各给出简短回答：\n{items}",
]

# All domains pooled for sampling
ALL_FACTS = GEOGRAPHY + SCIENCE + HISTORY + CULTURE + BIOLOGY
