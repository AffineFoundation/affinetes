"""
City Knowledge Graph for QQR Travel Planning.

Provides semantically grounded city profiles covering all 71 MAJOR_CITIES.
Used by problem_generator.py to ensure:
- Cities match their food themes (no seafood in landlocked cities)
- Interests align with city strengths
- Seasonal appropriateness (no winter trips to closed scenic areas)
- POI pool auto-generated from landmarks
"""

from dataclasses import dataclass, field
from typing import Dict, List

from config import MAJOR_CITIES


@dataclass
class CityProfile:
    specialties: List[str]      # City strengths (from MULTIDAY_INTERESTS)
    landmarks: List[str]        # Famous attractions (2-6)
    food_themes: List[str]      # Matching FOOD_THEMES (at least 1)
    seasonal_avoid: List[int]   # Months to avoid (empty = year-round)


CITY_KNOWLEDGE: Dict[str, CityProfile] = {
    # ======================== Tier 1 ========================
    "北京": CityProfile(
        specialties=["文化历史", "博物馆", "美食探店", "购物血拼"],
        landmarks=["故宫", "天坛", "颐和园", "长城", "南锣鼓巷", "什刹海"],
        food_themes=["地方特色小吃", "老字号探访", "夜市美食"],
        seasonal_avoid=[],
    ),
    "上海": CityProfile(
        specialties=["购物血拼", "美食探店", "文化历史", "博物馆"],
        landmarks=["外滩", "东方明珠", "豫园", "田子坊", "武康路", "迪士尼"],
        food_themes=["米其林餐厅", "老字号探访", "特色饮品甜品", "街头小吃"],
        seasonal_avoid=[],
    ),
    "广州": CityProfile(
        specialties=["美食探店", "文化历史", "购物血拼"],
        landmarks=["广州塔", "沙面", "陈家祠", "白云山", "长隆"],
        food_themes=["早茶文化", "地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[],
    ),
    "深圳": CityProfile(
        specialties=["购物血拼", "休闲度假", "美食探店"],
        landmarks=["世界之窗", "大梅沙", "东部华侨城", "深圳湾公园"],
        food_themes=["海鲜大餐", "街头小吃", "米其林餐厅"],
        seasonal_avoid=[],
    ),
    # ======================== New Tier 1 ========================
    "杭州": CityProfile(
        specialties=["自然风光", "文化历史", "休闲度假", "美食探店"],
        landmarks=["西湖", "灵隐寺", "西溪湿地", "千岛湖", "宋城"],
        food_themes=["地方特色小吃", "老字号探访", "特色饮品甜品"],
        seasonal_avoid=[],
    ),
    "成都": CityProfile(
        specialties=["美食探店", "休闲度假", "文化历史", "民俗体验"],
        landmarks=["宽窄巷子", "锦里", "武侯祠", "大熊猫基地", "杜甫草堂"],
        food_themes=["火锅美食", "地方特色小吃", "街头小吃", "夜市美食"],
        seasonal_avoid=[],
    ),
    "西安": CityProfile(
        specialties=["文化历史", "美食探店", "博物馆", "民俗体验"],
        landmarks=["兵马俑", "大雁塔", "回民街", "华清宫", "古城墙"],
        food_themes=["地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[],
    ),
    "重庆": CityProfile(
        specialties=["美食探店", "自然风光", "文化历史", "摄影打卡"],
        landmarks=["洪崖洞", "解放碑", "磁器口", "长江索道", "武隆天坑"],
        food_themes=["火锅美食", "地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[],
    ),
    "南京": CityProfile(
        specialties=["文化历史", "博物馆", "美食探店"],
        landmarks=["中山陵", "夫子庙", "明孝陵", "总统府", "玄武湖"],
        food_themes=["地方特色小吃", "老字号探访", "夜市美食"],
        seasonal_avoid=[],
    ),
    "武汉": CityProfile(
        specialties=["文化历史", "美食探店", "博物馆"],
        landmarks=["黄鹤楼", "东湖", "户部巷", "武汉大学", "长江大桥"],
        food_themes=["地方特色小吃", "街头小吃", "夜市美食"],
        seasonal_avoid=[7, 8],  # extreme heat
    ),
    "长沙": CityProfile(
        specialties=["美食探店", "文化历史", "休闲度假"],
        landmarks=["橘子洲", "岳麓山", "太平街", "湖南省博物馆"],
        food_themes=["地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[7, 8],
    ),
    "苏州": CityProfile(
        specialties=["文化历史", "古镇古村", "自然风光", "美食探店"],
        landmarks=["拙政园", "虎丘", "周庄", "金鸡湖", "平江路"],
        food_themes=["地方特色小吃", "老字号探访", "特色饮品甜品"],
        seasonal_avoid=[],
    ),
    "天津": CityProfile(
        specialties=["文化历史", "美食探店", "民俗体验"],
        landmarks=["五大道", "天津之眼", "古文化街", "瓷房子"],
        food_themes=["地方特色小吃", "老字号探访", "街头小吃"],
        seasonal_avoid=[],
    ),
    # ======================== Tier 2 ========================
    "青岛": CityProfile(
        specialties=["海滨度假", "美食探店", "休闲度假", "摄影打卡"],
        landmarks=["栈桥", "八大关", "崂山", "金沙滩", "啤酒博物馆"],
        food_themes=["海鲜大餐", "地方特色小吃", "特色饮品甜品"],
        seasonal_avoid=[],
    ),
    "厦门": CityProfile(
        specialties=["海滨度假", "文化历史", "美食探店", "摄影打卡"],
        landmarks=["鼓浪屿", "南普陀寺", "厦门大学", "环岛路", "曾厝垵"],
        food_themes=["海鲜大餐", "地方特色小吃", "街头小吃", "特色饮品甜品"],
        seasonal_avoid=[],
    ),
    "大连": CityProfile(
        specialties=["海滨度假", "自然风光", "休闲度假"],
        landmarks=["星海广场", "老虎滩", "金石滩", "棒棰岛"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "郑州": CityProfile(
        specialties=["文化历史", "博物馆", "美食探店"],
        landmarks=["少林寺", "河南博物院", "二七纪念塔", "黄河风景区"],
        food_themes=["地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[],
    ),
    "济南": CityProfile(
        specialties=["文化历史", "自然风光", "美食探店"],
        landmarks=["趵突泉", "大明湖", "千佛山", "芙蓉街"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "沈阳": CityProfile(
        specialties=["文化历史", "博物馆", "美食探店"],
        landmarks=["沈阳故宫", "张氏帅府", "北陵公园", "中街"],
        food_themes=["地方特色小吃", "老字号探访", "夜市美食"],
        seasonal_avoid=[12, 1, 2],
    ),
    "哈尔滨": CityProfile(
        specialties=["文化历史", "民俗体验", "摄影打卡"],
        landmarks=["中央大街", "圣索菲亚大教堂", "冰雪大世界", "太阳岛"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],  # winter is actually peak season for ice festival
    ),
    "福州": CityProfile(
        specialties=["文化历史", "美食探店", "自然风光"],
        landmarks=["三坊七巷", "鼓山", "西湖公园", "乌塔"],
        food_themes=["地方特色小吃", "海鲜大餐", "街头小吃"],
        seasonal_avoid=[7, 8, 9],  # typhoon season
    ),
    "合肥": CityProfile(
        specialties=["文化历史", "博物馆", "自然风光"],
        landmarks=["逍遥津", "包公祠", "安徽博物院", "巢湖"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "南昌": CityProfile(
        specialties=["文化历史", "自然风光", "美食探店"],
        landmarks=["滕王阁", "八一广场", "秋水广场", "梅岭"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[7, 8],
    ),
    "贵阳": CityProfile(
        specialties=["美食探店", "自然风光", "民俗体验"],
        landmarks=["黔灵山", "甲秀楼", "青岩古镇", "花溪公园"],
        food_themes=["地方特色小吃", "街头小吃", "夜市美食"],
        seasonal_avoid=[],
    ),
    "南宁": CityProfile(
        specialties=["自然风光", "美食探店", "民俗体验"],
        landmarks=["青秀山", "中山路", "南湖公园", "广西民族博物馆"],
        food_themes=["地方特色小吃", "夜市美食", "街头小吃"],
        seasonal_avoid=[],
    ),
    "海口": CityProfile(
        specialties=["海滨度假", "休闲度假", "美食探店"],
        landmarks=["骑楼老街", "假日海滩", "火山口公园", "万绿园"],
        food_themes=["海鲜大餐", "地方特色小吃", "街头小吃"],
        seasonal_avoid=[7, 8, 9],
    ),
    "太原": CityProfile(
        specialties=["文化历史", "美食探店", "博物馆"],
        landmarks=["晋祠", "山西博物院", "柳巷", "双塔寺"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "兰州": CityProfile(
        specialties=["美食探店", "文化历史", "自然风光"],
        landmarks=["黄河铁桥", "白塔山", "甘肃省博物馆", "黄河母亲雕塑"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[],
    ),
    "石家庄": CityProfile(
        specialties=["文化历史", "博物馆"],
        landmarks=["赵州桥", "隆兴寺", "河北博物院", "西柏坡"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "长春": CityProfile(
        specialties=["文化历史", "博物馆", "自然风光"],
        landmarks=["伪满皇宫", "净月潭", "长影世纪城", "南湖公园"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[12, 1, 2],
    ),
    "昆明": CityProfile(
        specialties=["自然风光", "休闲度假", "民俗体验", "美食探店"],
        landmarks=["石林", "滇池", "翠湖", "西山龙门", "云南民族村"],
        food_themes=["地方特色小吃", "街头小吃", "特色饮品甜品"],
        seasonal_avoid=[],
    ),
    "宁波": CityProfile(
        specialties=["文化历史", "海滨度假", "美食探店"],
        landmarks=["天一阁", "老外滩", "东钱湖", "溪口雪窦山"],
        food_themes=["海鲜大餐", "地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "无锡": CityProfile(
        specialties=["自然风光", "文化历史", "休闲度假"],
        landmarks=["太湖", "灵山大佛", "鼋头渚", "三国城", "惠山古镇"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "温州": CityProfile(
        specialties=["自然风光", "美食探店"],
        landmarks=["雁荡山", "楠溪江", "江心屿", "五马街"],
        food_themes=["海鲜大餐", "地方特色小吃", "街头小吃"],
        seasonal_avoid=[7, 8, 9],
    ),
    "烟台": CityProfile(
        specialties=["海滨度假", "自然风光", "休闲度假"],
        landmarks=["蓬莱阁", "长岛", "烟台山", "养马岛"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[],
    ),
    # ======================== Tourist Cities ========================
    "三亚": CityProfile(
        specialties=["海滨度假", "自然风光", "休闲度假"],
        landmarks=["亚龙湾", "天涯海角", "蜈支洲岛", "南山寺"],
        food_themes=["海鲜大餐"],
        seasonal_avoid=[7, 8, 9],  # typhoon season
    ),
    "桂林": CityProfile(
        specialties=["自然风光", "摄影打卡", "休闲度假"],
        landmarks=["漓江", "象鼻山", "阳朔西街", "龙脊梯田"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[],
    ),
    "丽江": CityProfile(
        specialties=["自然风光", "文化历史", "民俗体验", "休闲度假"],
        landmarks=["丽江古城", "玉龙雪山", "束河古镇", "泸沽湖"],
        food_themes=["地方特色小吃", "民族风味"],
        seasonal_avoid=[],
    ),
    "张家界": CityProfile(
        specialties=["自然风光", "户外运动", "摄影打卡", "登山徒步"],
        landmarks=["张家界国家森林公园", "天门山", "玻璃桥", "宝峰湖"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    "黄山": CityProfile(
        specialties=["自然风光", "登山徒步", "摄影打卡"],
        landmarks=["黄山", "宏村", "西递", "屯溪老街"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "九寨沟": CityProfile(
        specialties=["自然风光", "摄影打卡"],
        landmarks=["九寨沟", "黄龙"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],  # winter closure
    ),
    "洛阳": CityProfile(
        specialties=["文化历史", "博物馆", "赏花观鸟"],
        landmarks=["龙门石窟", "白马寺", "洛阳博物馆", "关林"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "泉州": CityProfile(
        specialties=["文化历史", "民俗体验", "美食探店"],
        landmarks=["开元寺", "清净寺", "洛阳桥", "崇武古城"],
        food_themes=["地方特色小吃", "海鲜大餐", "街头小吃"],
        seasonal_avoid=[7, 8, 9],
    ),
    "大理": CityProfile(
        specialties=["自然风光", "休闲度假", "民俗体验", "摄影打卡"],
        landmarks=["洱海", "苍山", "大理古城", "崇圣寺三塔"],
        food_themes=["地方特色小吃", "素食禅意料理"],
        seasonal_avoid=[],
    ),
    "威海": CityProfile(
        specialties=["海滨度假", "自然风光", "休闲度假"],
        landmarks=["刘公岛", "成山头", "那香海", "威海国际海水浴场"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[],
    ),
    "珠海": CityProfile(
        specialties=["海滨度假", "休闲度假"],
        landmarks=["长隆海洋王国", "情侣路", "圆明新园", "外伶仃岛"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[7, 8, 9],
    ),
    "北海": CityProfile(
        specialties=["海滨度假", "自然风光", "休闲度假"],
        landmarks=["北海银滩", "涠洲岛", "北海老街", "红树林"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[7, 8, 9],
    ),
    "秦皇岛": CityProfile(
        specialties=["海滨度假", "文化历史", "休闲度假"],
        landmarks=["北戴河", "山海关", "老龙头", "鸽子窝公园"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "敦煌": CityProfile(
        specialties=["文化历史", "自然风光", "摄影打卡"],
        landmarks=["莫高窟", "鸣沙山月牙泉", "雅丹地貌", "玉门关"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "景德镇": CityProfile(
        specialties=["文化历史", "民俗体验", "博物馆"],
        landmarks=["景德镇古窑", "陶瓷博物馆", "御窑厂", "三宝蓬"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    "婺源": CityProfile(
        specialties=["自然风光", "古镇古村", "摄影打卡", "赏花观鸟"],
        landmarks=["篁岭", "江岭", "李坑", "思溪延村"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    "平遥": CityProfile(
        specialties=["文化历史", "古镇古村", "民俗体验"],
        landmarks=["平遥古城", "日升昌票号", "城隍庙", "镇国寺"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    # ======================== Sichuan Corridor ========================
    "乐山": CityProfile(
        specialties=["自然风光", "文化历史", "美食探店"],
        landmarks=["乐山大佛", "峨眉山", "嘉定坊"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[],
    ),
    "都江堰": CityProfile(
        specialties=["文化历史", "自然风光"],
        landmarks=["都江堰", "青城山", "灌县古城"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    "峨眉山": CityProfile(
        specialties=["自然风光", "登山徒步", "文化历史"],
        landmarks=["峨眉山", "金顶", "万年寺", "报国寺"],
        food_themes=["地方特色小吃", "素食禅意料理"],
        seasonal_avoid=[],
    ),
    "稻城": CityProfile(
        specialties=["自然风光", "摄影打卡", "户外运动", "登山徒步"],
        landmarks=["亚丁", "牛奶海", "五色海", "仙乃日"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2, 3],  # winter/early spring closure + altitude
    ),
    # ======================== Western Cities ========================
    "银川": CityProfile(
        specialties=["文化历史", "自然风光", "民俗体验"],
        landmarks=["西夏王陵", "镇北堡西部影城", "沙湖", "贺兰山岩画"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "西宁": CityProfile(
        specialties=["自然风光", "文化历史", "民俗体验"],
        landmarks=["青海湖", "塔尔寺", "茶卡盐湖", "东关清真大寺"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "呼和浩特": CityProfile(
        specialties=["自然风光", "民俗体验", "文化历史"],
        landmarks=["大召寺", "内蒙古博物院", "昭君墓", "敕勒川草原"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "乌鲁木齐": CityProfile(
        specialties=["自然风光", "民俗体验", "美食探店"],
        landmarks=["天山天池", "新疆国际大巴扎", "红山公园", "南山牧场"],
        food_themes=["地方特色小吃", "街头小吃", "夜市美食"],
        seasonal_avoid=[12, 1, 2],
    ),
    "拉萨": CityProfile(
        specialties=["文化历史", "自然风光", "民俗体验", "摄影打卡"],
        landmarks=["布达拉宫", "大昭寺", "八廓街", "纳木错", "罗布林卡"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    # ======================== Jiangsu/Zhejiang Extended ========================
    "常州": CityProfile(
        specialties=["休闲度假", "亲子游乐", "文化历史"],
        landmarks=["恐龙园", "天目湖", "淹城春秋乐园", "南山竹海"],
        food_themes=["地方特色小吃", "老字号探访"],
        seasonal_avoid=[],
    ),
    "徐州": CityProfile(
        specialties=["文化历史", "博物馆"],
        landmarks=["云龙湖", "汉文化景区", "徐州博物馆", "龟山汉墓"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    "连云港": CityProfile(
        specialties=["海滨度假", "自然风光", "文化历史"],
        landmarks=["花果山", "连岛", "海州古城", "渔湾"],
        food_themes=["海鲜大餐", "地方特色小吃"],
        seasonal_avoid=[],
    ),
    "扬州": CityProfile(
        specialties=["文化历史", "美食探店", "自然风光"],
        landmarks=["瘦西湖", "个园", "何园", "东关街"],
        food_themes=["地方特色小吃", "老字号探访", "早茶文化"],
        seasonal_avoid=[],
    ),
    # ======================== Yunnan Extended ========================
    "腾冲": CityProfile(
        specialties=["自然风光", "温泉养生", "文化历史"],
        landmarks=["和顺古镇", "热海温泉", "银杏村", "火山地热"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[7, 8],  # rainy season
    ),
    # ======================== Guangxi Extended ========================
    "阳朔": CityProfile(
        specialties=["自然风光", "户外运动", "休闲度假", "摄影打卡"],
        landmarks=["十里画廊", "遇龙河", "月亮山", "西街"],
        food_themes=["地方特色小吃", "街头小吃"],
        seasonal_avoid=[],
    ),
    # ======================== Hunan Extended ========================
    "凤凰古城": CityProfile(
        specialties=["文化历史", "古镇古村", "民俗体验", "摄影打卡"],
        landmarks=["凤凰古城", "沱江", "虹桥", "南华山"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[],
    ),
    # ======================== Gansu Extended ========================
    "张掖": CityProfile(
        specialties=["自然风光", "摄影打卡"],
        landmarks=["七彩丹霞", "张掖大佛寺", "平山湖大峡谷"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    "嘉峪关": CityProfile(
        specialties=["文化历史", "自然风光"],
        landmarks=["嘉峪关关城", "悬壁长城", "天下第一墩"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
    # ======================== Hebei Extended ========================
    "承德": CityProfile(
        specialties=["文化历史", "自然风光", "休闲度假"],
        landmarks=["避暑山庄", "外八庙", "金山岭长城", "塞罕坝"],
        food_themes=["地方特色小吃"],
        seasonal_avoid=[12, 1, 2],
    ),
}


# ============================================================================
# Module-level validation: every MAJOR_CITIES entry must have a profile
# ============================================================================
_missing = set(MAJOR_CITIES) - set(CITY_KNOWLEDGE.keys())
assert not _missing, f"Knowledge graph missing cities: {_missing}"

_extra = set(CITY_KNOWLEDGE.keys()) - set(MAJOR_CITIES)
assert not _extra, f"Knowledge graph has extra cities not in MAJOR_CITIES: {_extra}"


# ============================================================================
# Helper functions
# ============================================================================

def get_profile(city: str) -> CityProfile:
    """Get city profile. All MAJOR_CITIES are guaranteed to exist."""
    return CITY_KNOWLEDGE[city]


def cities_for_food_theme(theme: str) -> List[str]:
    """Return cities matching the given food theme."""
    return [
        city for city, profile in CITY_KNOWLEDGE.items()
        if theme in profile.food_themes
    ]


def is_season_ok(city: str, month: int) -> bool:
    """Check if the given month is appropriate for visiting the city."""
    profile = CITY_KNOWLEDGE.get(city)
    if profile is None:
        return True
    return month not in profile.seasonal_avoid


def get_all_landmarks() -> Dict[str, List[str]]:
    """Return {landmark: [cities]} mapping for POI_CITY_MAP generation."""
    result: Dict[str, List[str]] = {}
    for city, profile in CITY_KNOWLEDGE.items():
        for landmark in profile.landmarks:
            if landmark not in result:
                result[landmark] = []
            result[landmark].append(city)
    return result
