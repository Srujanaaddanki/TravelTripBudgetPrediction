import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pages.plan_trip import get_ml_proxy_destination, _resolve_destination
from src.services.confidence_engine import ConfidenceEngine
from src.services.travel_intelligence import TravelIntelligenceEngine


class TestHybridDestinationIntelligence(unittest.TestCase):
    def test_ml_proxy_destination_mappings(self):
        """Test proxy mapping helper handles specific profiles and fallback rules."""
        # Check specific exact overrides
        self.assertEqual(get_ml_proxy_destination("Annavaram"), "tirupati")
        self.assertEqual(get_ml_proxy_destination("Kukanet"), "kukunate")
        self.assertEqual(get_ml_proxy_destination("Paris"), "dubai")
        self.assertEqual(get_ml_proxy_destination("London", country="United Kingdom"), "dubai")

        # Check categorical checks
        # Temple & high altitude
        self.assertEqual(
            get_ml_proxy_destination("Some High Temple", tourism_category="Temple", altitude=3000), 
            "kedarnath"
        )
        # Beach
        self.assertEqual(get_ml_proxy_destination("Some Beach", tourism_category="Beach"), "goa")
        # High altitude hills
        self.assertEqual(
            get_ml_proxy_destination("High Peak", tourism_category="Hill Station", altitude=2800), 
            "spiti"
        )
        # Regional fallbacks
        self.assertEqual(get_ml_proxy_destination("Unresolved Andhra Place", state="Andhra Pradesh"), "vijayawada")
        self.assertEqual(get_ml_proxy_destination("Unresolved Punjab Place", state="Punjab"), "jalandhar")
        self.assertEqual(get_ml_proxy_destination("Unknown Place"), "delhi")

    @patch("src.pages.plan_trip._geo_service")
    @patch("src.pages.plan_trip._dest_cache")
    def test_resolve_destination_pipeline(self, mock_cache, mock_geo):
        """Test the multi-step destination resolution pipeline."""
        mock_cache.get_intelligence_cache.return_value = None
        mock_cache.get_cached.return_value = None
        
        # Test Exact match
        encoder_classes = ["Goa", "Tirupati", "Delhi", "Manali", "Kukunate"]
        res = _resolve_destination("Goa", encoder_classes)
        self.assertTrue(res["is_known"])
        self.assertEqual(res["matched_dest"], "goa")
        self.assertEqual(res["match_tier"], "exact")

        # Test Geoapify geocoding success (Unknown destination)
        mock_geo.validate_destination.return_value = {
            "valid": True,
            "lat": 17.06,
            "lng": 82.40,
            "country": "India",
            "state": "Andhra Pradesh",
            "display_name": "Annavaram, Andhra Pradesh"
        }
        
        # Mock Gemini metadata classifier response
        with patch("src.services.gemini_service.GeminiService.resolve_unknown_destination_metadata") as mock_meta:
            mock_meta.return_value = {
                "country": "India",
                "state": "Andhra Pradesh",
                "altitude": 100.0,
                "tourism_category": "Temple",
                "weather_profile": "tropical"
            }
            
            res_unknown = _resolve_destination("Annavaram", encoder_classes)
            self.assertFalse(res_unknown["is_known"])
            # Proxy mapping should map Annavaram/Temple/Andhra to tirupati
            self.assertEqual(res_unknown["matched_dest"], "tirupati")
            self.assertEqual(res_unknown["match_tier"], "geo_validated")
            self.assertEqual(res_unknown["dst_coords"], (17.06, 82.40))

    def test_confidence_engine_scaling(self):
        """Verify confidence scores match requirements for known, API only, AI and fallback resolutions."""
        engine = ConfidenceEngine()
        
        # Known destination
        res_known = engine.calculate_confidence(
            dataset_insights={"has_data": True, "similar_count": 6},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=True,
            resolution_type="known"
        )
        self.assertEqual(res_known["score"], 100)
        self.assertEqual(res_known["level"], "Dataset Verified")

        # API estimated
        res_api = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=False,
            resolution_type="api_only"
        )
        self.assertEqual(res_api["score"], 80)
        self.assertEqual(res_api["level"], "API Estimated")

        # AI approximation (Gemini)
        res_gemini = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 500},
            ml_prediction=15000,
            is_known=False,
            resolution_type="gemini_approx"
        )
        self.assertEqual(res_gemini["score"], 65)
        self.assertEqual(res_gemini["level"], "AI Approximation")

        # Fallback
        res_fallback = engine.calculate_confidence(
            dataset_insights={"has_data": False, "similar_count": 0},
            mode_comparison={"distance_km": 0},
            ml_prediction=15000,
            is_known=False,
            resolution_type="failed"
        )
        self.assertEqual(res_fallback["score"], 50)
        self.assertEqual(res_fallback["level"], "Low Reliability (Failed Resolution)")


if __name__ == "__main__":
    unittest.main()
