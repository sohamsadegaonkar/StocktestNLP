"""Virtual monochrome display. The generated PNG uses exactly 1 bit per pixel."""
from io import BytesIO

from PIL import Image, ImageDraw


def display_png(state: dict) -> bytes:
    display = Image.new("1", (128, 64))
    draw = ImageDraw.Draw(display)
    draw.rectangle((0, 0, 127, 13), fill=1)
    draw.text((4, 0), "StockBrief Edge", fill=0)
    link = "UP" if state["connected"] else "DOWN"
    draw.text((4, 16), f"Link {link}  Queue {state['pending']}", fill=1)
    draw.text((4, 31), f"Packet {state['cursor']}/{state['total_packets']}", fill=1)
    draw.text((4, 46), "Local inference ON", fill=1)
    output = BytesIO()
    display.save(output, format="PNG")
    return output.getvalue()
