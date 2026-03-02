"""
Output Parser - Parse model output into structured data for scoring.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TransportOption:
    """Transport option (flight/train)."""
    type: str = ""
    number: str = ""
    price: float = 0.0
    departure_time: str = ""
    arrival_time: str = ""
    duration: str = ""
    from_station: str = ""
    to_station: str = ""


@dataclass
class ScheduleItem:
    """Schedule item in daily plan."""
    time_start: str = "00:00"
    time_end: str = "00:00"
    location: str = ""
    type: str = ""
    activity: str = ""
    cost: float = 0.0
    def to_dict(self) -> dict:
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "location": self.location,
            "type": self.type,
            "activity": self.activity,
            "cost": self.cost,
        }


@dataclass
class DayPlan:
    """Daily plan."""
    day: int = 0
    date: str = ""
    theme: str = ""
    items: List[ScheduleItem] = field(default_factory=list)
    day_total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "date": self.date,
            "theme": self.theme,
            "items": [item.to_dict() for item in self.items],
            "day_total": self.day_total,
        }


@dataclass
class ParsedOutput:
    """Parsed output structure."""
    summary: Dict[str, Any] = field(default_factory=dict)
    transport_options: List[TransportOption] = field(default_factory=list)
    recommended_transport: Optional[TransportOption] = None
    daily_plans: List[DayPlan] = field(default_factory=list)
    budget_breakdown: Dict[str, float] = field(default_factory=dict)
    tips: List[str] = field(default_factory=list)
    weather_info: Dict[str, Any] = field(default_factory=dict)
    mentioned_locations: List[str] = field(default_factory=list)
    raw_text: str = ""
    parse_success: bool = False
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "transport_options": [
                {"type": t.type, "number": t.number, "price": t.price}
                for t in self.transport_options
            ],
            "recommended_transport": (
                {"type": self.recommended_transport.type, "number": self.recommended_transport.number}
                if self.recommended_transport else None
            ),
            "daily_plans": [day.to_dict() for day in self.daily_plans],
            "budget_breakdown": self.budget_breakdown,
            "tips": self.tips,
            "weather_info": self.weather_info,
            "mentioned_locations": self.mentioned_locations,
            "parse_success": self.parse_success,
            "parse_errors": self.parse_errors,
        }

class OutputParser:
    """Output parser for model responses."""

    def parse(self, raw_output: str) -> ParsedOutput:
        """Parse model output."""
        result = ParsedOutput(raw_text=raw_output)
        errors = []

        if not raw_output or len(raw_output.strip()) < 50:
            result.parse_success = False
            result.parse_errors = ["Output too short or empty"]
            return result

        json_data = self._extract_json(raw_output)

        if json_data:
            try:
                self._parse_json_output(result, json_data)
                result.parse_success = True
            except Exception as e:
                errors.append(f"JSON parse error: {str(e)}")
                self._parse_text_output(result, raw_output)
        else:
            self._parse_text_output(result, raw_output)
            result.parse_success = len(result.daily_plans) > 0 or len(result.transport_options) > 0

        result.mentioned_locations = self._extract_locations(raw_output)
        result.parse_errors = errors
        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from text."""
        json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_block_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        code_block_pattern = r'```\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        brace_start = text.find('{')
        if brace_start != -1:
            depth = 0
            for i, c in enumerate(text[brace_start:]):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:brace_start + i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def _parse_json_output(self, result: ParsedOutput, data: dict):
        """Parse JSON format output."""
        result.summary = data.get("summary", {})

        if "transport_options" in data or "flights" in data or "trains" in data:
            transport_data = data.get("transport_options", [])
            transport_data.extend(data.get("flights", []))
            transport_data.extend(data.get("trains", []))

            for t in transport_data:
                option = TransportOption(
                    type=t.get("type", ""),
                    number=t.get("number", t.get("flight_no", t.get("train_no", ""))),
                    price=float(t.get("price", 0)),
                    departure_time=t.get("departure_time", t.get("dep_time", "")),
                    arrival_time=t.get("arrival_time", t.get("arr_time", "")),
                    duration=t.get("duration", ""),
                )
                result.transport_options.append(option)

        if "recommended_transport" in data:
            rec = data["recommended_transport"]
            result.recommended_transport = TransportOption(
                type=rec.get("type", ""),
                number=rec.get("number", ""),
                price=float(rec.get("price", 0)),
            )

        result.daily_plans = self._parse_daily_plans(data.get("daily_plans", []))
        result.budget_breakdown = data.get("budget_breakdown", data.get("budget", {}))
        result.tips = data.get("tips", [])
        result.weather_info = data.get("weather", data.get("weather_info", {}))

    def _parse_daily_plans(self, plans_data: List[dict]) -> List[DayPlan]:
        """Parse daily plans from JSON."""
        daily_plans = []

        for plan_data in plans_data:
            items = []
            for item_data in plan_data.get("items", plan_data.get("schedule", [])):
                item = self._parse_schedule_item(item_data)
                items.append(item)

            day_plan = DayPlan(
                day=plan_data.get("day", 0),
                date=plan_data.get("date", ""),
                theme=plan_data.get("theme", ""),
                items=items,
                day_total=float(plan_data.get("day_total", plan_data.get("total", 0))),
            )
            daily_plans.append(day_plan)

        return daily_plans

    def _parse_schedule_item(self, item_data: dict) -> ScheduleItem:
        """Parse single schedule item."""
        time_str = item_data.get("time", "00:00-00:00")
        time_start, time_end = self._parse_time_range(time_str)

        return ScheduleItem(
            time_start=time_start,
            time_end=time_end,
            location=item_data.get("location", item_data.get("place", "")),
            type=item_data.get("type", ""),
            activity=item_data.get("activity", item_data.get("description", "")),
            cost=float(item_data.get("cost", item_data.get("price", 0))),
        )

    def _parse_time_range(self, time_str: str) -> tuple:
        """Parse time range string."""
        if not time_str:
            return "00:00", "00:00"

        parts = re.split(r'[-~—至到]', time_str)
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()

        return time_str.strip(), "00:00"

    def _parse_text_output(self, result: ParsedOutput, text: str):
        """Parse from plain text."""
        result.transport_options = self._extract_transport_from_text(text)
        result.daily_plans = self._extract_days_from_text(text)
        result.budget_breakdown = self._extract_budget_from_text(text)

    def _extract_transport_from_text(self, text: str) -> List[TransportOption]:
        """Extract transport info from text."""
        options = []

        flight_pattern = r'航班\s*([A-Z]{2}\d{3,4})[，,].*?价格\s*(\d+(?:\.\d+)?)\s*元'
        for match in re.finditer(flight_pattern, text):
            options.append(TransportOption(
                type="flight",
                number=match.group(1),
                price=float(match.group(2)),
            ))

        train_pattern = r'(?:车次|列车)\s*([GDCZTK]\d+)[，,].*?价格\s*(\d+(?:\.\d+)?)\s*元'
        for match in re.finditer(train_pattern, text):
            options.append(TransportOption(
                type="train",
                number=match.group(1),
                price=float(match.group(2)),
            ))

        return options

    def _extract_days_from_text(self, text: str) -> List[DayPlan]:
        """Extract daily plans from text."""
        daily_plans = []

        day_pattern = r'(?:第\s*(\d+)\s*天|Day\s*(\d+))'
        day_matches = list(re.finditer(day_pattern, text, re.IGNORECASE))

        if not day_matches:
            return []

        for i, match in enumerate(day_matches):
            day_num = int(match.group(1) or match.group(2))

            start_pos = match.end()
            end_pos = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(text)
            day_content = text[start_pos:end_pos]

            items = self._extract_items_from_text(day_content)

            daily_plans.append(DayPlan(
                day=day_num,
                items=items,
                day_total=sum(item.cost for item in items),
            ))

        return daily_plans

    def _extract_items_from_text(self, text: str) -> List[ScheduleItem]:
        """Extract schedule items from text paragraph."""
        items = []

        time_location_pattern = r'(\d{1,2}:\d{2})\s*[-~—至到]\s*(\d{1,2}:\d{2})\s*[：:]*\s*([^\n\d]{2,30})'
        for match in re.finditer(time_location_pattern, text):
            items.append(ScheduleItem(
                time_start=match.group(1),
                time_end=match.group(2),
                location=match.group(3).strip(),
                type="景点",
                activity="游览",
            ))

        if not items:
            location_patterns = [
                r'【([^】]+)】',
                r'「([^」]+)」',
                r'(?:游览|参观|前往)\s*([^\n，。,\.]{2,20})',
            ]
            for pattern in location_patterns:
                for match in re.finditer(pattern, text):
                    location = match.group(1).strip()
                    if location and len(location) > 1:
                        items.append(ScheduleItem(
                            location=location,
                            type="景点",
                            activity="游览",
                        ))

        return items

    def _extract_budget_from_text(self, text: str) -> Dict[str, float]:
        """Extract budget info from text."""
        budget = {}

        categories = {
            "交通": ["交通", "机票", "火车票", "车费"],
            "住宿": ["住宿", "酒店", "房费"],
            "餐饮": ["餐饮", "吃饭", "用餐", "美食"],
            "门票": ["门票", "景点", "票价"],
            "其他": ["其他", "杂费", "购物"],
            "总计": ["总计", "合计", "总共", "总预算"],
        }

        for category, keywords in categories.items():
            for kw in keywords:
                pattern = rf'{kw}[：:]*\s*(\d+(?:\.\d+)?)\s*元?'
                match = re.search(pattern, text)
                if match:
                    budget[category] = float(match.group(1))
                    break

        return budget

    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        locations = []

        suffixes = [
            "景区", "公园", "博物馆", "故宫", "寺", "庙", "塔", "山", "湖", "海",
            "广场", "街", "路", "站", "机场", "酒店", "餐厅", "商场", "中心",
            "古镇", "老街", "风景区", "遗址", "纪念馆", "美术馆", "大学",
        ]

        suffix_pattern = "|".join(re.escape(s) for s in suffixes)
        pattern = rf'[^\n，。,\.\s]{{2,10}}(?:{suffix_pattern})'

        for match in re.finditer(pattern, text):
            location = match.group().strip()
            if location not in locations:
                locations.append(location)

        return locations


_default_parser = None


def get_parser() -> OutputParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = OutputParser()
    return _default_parser
