import time
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write
import adafruit_pixelbuf

# NeoPixel Configuration
NEOPIXEL_PIN = board.D13  # GPIO pin connected to NeoPixels
NUM_PIXELS = 12           # 12 LEDs for Adafruit NeoPixel X12 ring
BRIGHTNESS = 0.5          # Initial brightness (0.0 to 1.0)

class NeoPixelRing(adafruit_pixelbuf.PixelBuf):
    """Custom NeoPixel class for Raspberry Pi 5"""
    def __init__(self, pin, n, brightness=1.0, **kwargs):
        self._pin = pin
        self._brightness = brightness
        super().__init__(size=n, **kwargs)

    def _transmit(self, buf):
        """Send buffer to NeoPixels with brightness adjustment"""
        if self._brightness < 1.0:
            buf = bytearray(int(val * self._brightness) for val in buf)
        neopixel_write(self._pin, buf)

    @property
    def brightness(self):
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        """Set brightness (0.0 to 1.0)"""
        self._brightness = max(0.0, min(1.0, value))

def set_white(pixels, brightness=None):
    """Set all pixels to white with optional brightness"""
    if brightness is not None:
        pixels.brightness = brightness
    white = (255, 255, 255)
    pixels.fill(white)
    pixels.show()

# Initialize NeoPixel ring
pixels = NeoPixelRing(
    NEOPIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=False,  # Better for brightness control
    pixel_order="GRB"  # Common NeoPixel order
)

try:
    # Set initial white light
    set_white(pixels)
    
    # Example brightness adjustment (press Ctrl+C to exit)
    print("NeoPixel X12 Ring - White Light")
    print("Current brightness:", pixels.brightness)
    
    while True:
        # Add your brightness control logic here
        # Example: Gradually change brightness
        for b in [0.1, 0.3, 0.7, 1.0, 0.5, 0.2]:
            set_white(pixels, brightness=b)
            print(f"Brightness set to: {b*100:.0f}%")
            time.sleep(2)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    # Cleanup
    pixels.fill(0)
    pixels.show()
    print("LEDs turned off")