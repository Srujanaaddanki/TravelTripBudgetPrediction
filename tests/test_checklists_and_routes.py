import unittest
from src.intelligence.destination_rules import (
    get_destination_checklist,
    get_destination_country,
    get_destination_altitude,
    get_destination_permits_required,
    get_indirect_route_bars
)

class TestChecklistsAndRoutes(unittest.TestCase):
    def test_destination_country(self):
        self.assertEqual(get_destination_country("Kedarnath"), "India")
        self.assertEqual(get_destination_country("Paris, France"), "France")
        self.assertEqual(get_destination_country("Goa"), "India")
        self.assertEqual(get_destination_country("London"), "International")

    def test_destination_altitude(self):
        self.assertEqual(get_destination_altitude("Kedarnath"), 3500)
        self.assertEqual(get_destination_altitude("Goa"), 10)
        self.assertEqual(get_destination_altitude("Manali"), 1800)

    def test_destination_permits(self):
        self.assertTrue(get_destination_permits_required("Kedarnath"))
        self.assertFalse(get_destination_permits_required("Goa"))
        self.assertTrue(get_destination_permits_required("Amarnath"))

    def test_indirect_route_bars(self):
        bars = get_indirect_route_bars("Hyderabad", "Kedarnath")
        self.assertIsNotNone(bars)
        self.assertEqual(bars["Train"], "✓ Hyderabad → Haridwar")
        self.assertEqual(bars["Bus"], "✓ Haridwar → Sonprayag")
        self.assertEqual(bars["Flight"], "✓ Hyderabad → Dehradun")

        # Goa has direct access, should return None
        self.assertIsNone(get_indirect_route_bars("Hyderabad", "Goa"))

    def test_checklist_generation_kedarnath(self):
        checklist = get_destination_checklist("Kedarnath", month="January", travel_mode="Train", trip_type="Trekking")
        # Check high-altitude items
        self.assertIn("Altitude medicine", checklist["packing"])
        self.assertIn("Thermal wear", checklist["packing"])
        # Check trekking items
        self.assertIn("Trekking shoes", checklist["packing"])
        self.assertIn("Torch", checklist["packing"])
        # Check permit registration
        self.assertIn("Char Dham Registration", checklist["pretravel"])
        self.assertIn("Medical Fitness Certificate", checklist["pretravel"])
        # Check mode-specific items
        self.assertIn("Train tickets", checklist["packing"])

    def test_checklist_generation_tirupati(self):
        checklist = get_destination_checklist("Tirupati", month="January", travel_mode="Car", trip_type="Religious")
        # Check temple specific items
        self.assertIn("Temple dress code", checklist["packing"])
        self.assertIn("ID proof", checklist["packing"])
        self.assertIn("Darshan ticket booking", checklist["pretravel"])
        self.assertIn("Special entry ticket", checklist["pretravel"])
        self.assertIn("Cash for offerings", checklist["pretravel"])

    def test_checklist_generation_international(self):
        checklist = get_destination_checklist("Paris", month="July", travel_mode="Flight", country="France")
        # Check international items
        self.assertIn("Passport (valid 6 months)", checklist["packing"])
        self.assertIn("Visa / e-Visa", checklist["pretravel"])
        self.assertIn("Travel Insurance", checklist["pretravel"])
        # Check monsoon/rain items
        self.assertIn("Rain jacket", checklist["packing"])

if __name__ == "__main__":
    unittest.main()
