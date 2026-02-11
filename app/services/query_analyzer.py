"""Query analyzer service to extract filters from user queries using LLM"""

import json
import logging
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze user queries to extract search intent and filters"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def analyze_query(self, user_query: str) -> dict:
        """
        Analyze user query to extract:
        1. Clean search query
        2. Metadata filters (price range, categories, etc.)

        Args:
            user_query: Raw user query

        Returns:
            dict with 'search_query' and 'filters'
        """
        system_prompt = """You are a query analyzer for a furniture product database. Extract semantic search terms and metadata filters from natural language queries and return the result as JSON.

METADATA STRUCTURE:
{
  "product_code": "1004",
  "base_price": 1325.0,
  "description": "CHAP - Upholstered Round Back Armchair.",
  "categories": ["Wood Frame Seating", "Workplace", "Guest Seating", "Education"],
  "height_value": 30.0, "height_unit": "IN",
  "width_value": 24.5, "width_unit": "IN", 
  "depth_value": 22.5, "depth_unit": "IN",
  "weight_value": 26.0, "weight_unit": "LBS",
  "volume_value": 13.25, "volume_unit": "CUINCH",
  "series": "Chap"
}

FILTERABLE FIELDS:
- product_code (string): Unique product identifier
- base_price (float): Price in USD
- categories (list): List of category names
- height_value, width_value, depth_value, weight_value, volume_value (float): Product dimensions

CATEGORY KEYWORDS (18 categories - include in search_query for semantic matching):
- Benches and Ottomans
- Cafe and Cafeteria Seating
- Classroom Seating
- Conference and Management Seating
- Dining Cafeteria
- Education
- Guest Seating
- Healthcare
- Heavy Duty and 24HR Office Seating
- Lounge Seating
- Mesh Seating
- Pedestal Seating
- Stacking and Nesting Chairs
- Stools
- Tandem Seating
- Wood Frame Seating
- Work and Task Seating
- Workplace

FILTER OPERATORS:
- Numeric: {"$lte": X}, {"$gte": X}, {"$eq": X}, or {"$gte": X, "$lte": Y} for ranges
- String (categories/series): Single string or list of strings for multiple categories

OUTPUT FORMAT (JSON):
{
  "search_query": "core product attributes for semantic search",
  "filters": {"field": {"$op": value}} or {"field": "string"} or {"field": ["string1", "string2"]}
}

Return your response as a valid JSON object with these exact keys.

EXAMPLES:

Query: "guest seating"
{"search_query": "seating", "filters": {"categories": ["Guest Seating"]}}

Query: "workplace chair under $500"
{"search_query": "chair", "filters": {"categories": ["Workplace"], "base_price": {"$lte": 500}}}

Query: "education or workplace chair"
{"search_query": "chair", "filters": {"categories": ["Education", "Workplace"]}}

Query: "chair between $1100 and $1400"
{"search_query": "chair", "filters": {"base_price": {"$gte": 1100, "$lte": 1400}}}

Query: "mesh seating under $900"
{"search_query": "seating", "filters": {"categories": ["Mesh Seating"], "base_price": {"$lte": 900}}}

Query: "healthcare chair over 40 inches"
{"search_query": "chair", "filters": {"categories": ["Healthcare"], "height_value": {"$gte": 40}}}

Query: "comfortable chair"
{"search_query": "comfortable chair", "filters": {}}

CRITICAL RULES:
1. Extract product type/material/style as search_query (remove category keywords from search)
2. ALWAYS detect price ranges: "between $X and $Y", "from $X to $Y" → {"$gte": X, "$lte": Y}
3. ALWAYS map category keywords to official names and add to filters as list: {"categories": ["Category1", "Category2"]}
4. Category detection: map user terms to exact category names from the 18 categories list
5. Multiple categories: If "or" or multiple mentioned, list them: ["Workplace", "Education"]
6. Single category: Still use list format: ["Workplace"]
7. Price keywords: "under/below" → $lte, "over/above" → $gte, "between/from...to" → both
8. Normalize synonyms: "conference" → "Conference and Management Seating", "24hr" → "Heavy Duty and 24HR Office Seating"
9. Series names (Global Accord, Granada, Kate, Wind, Malaga, etc.) should be kept in search_query for semantic matching, NOT as filters
10. Return valid JSON with "search_query" and "filters" keys
"""

        try:
            response = self.client.chat.completions.create(
                model=settings.QUERY_ANALYZER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            
            # Ensure we have the expected structure
            if "search_query" not in result:
                result["search_query"] = user_query
            if "filters" not in result:
                result["filters"] = {}

            logger.info(f"Query analysis result: {result}")

            return result

        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            # Fallback: use original query with no filters
            return {
                "search_query": user_query,
                "filters": {}
            }


@lru_cache
def get_query_analyzer() -> QueryAnalyzer:
    """Get cached query analyzer instance"""
    return QueryAnalyzer()
