"""Image-only training label proposals; never used by policy inference."""
import cv2
import numpy as np

COLORS = ('red', 'green', 'blue')


def candidates(rgb):
    """Visible color components, not amodal object centers or simulator truth."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    chromatic = (s >= 100) & (v >= 45)
    hues = ((h <= 10) | (h >= 170), (h >= 35) & (h <= 85), (h >= 95) & (h <= 135))
    result = []
    for color, hue in zip(COLORS, hues):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (chromatic & hue).astype(np.uint8), connectivity=8)
        components = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
        components = [i for i in components if stats[i, cv2.CC_STAT_AREA] >= 5]
        row = dict(color=color, accepted=False, components=len(components), area=0, dominance=0.,
                   center_xy=None, box_xywh=None, touches_image_edge=False)
        if components:
            i = components[0]
            x, y, width, height, area = map(int, stats[i])
            dominance = float(area / sum(int(stats[j, cv2.CC_STAT_AREA]) for j in components))
            edge = x == 0 or y == 0 or x + width == rgb.shape[1] or y + height == rgb.shape[0]
            row.update(area=area, dominance=dominance, center_xy=centers[i].tolist(),
                       box_xywh=[x, y, width, height], touches_image_edge=edge,
                       accepted=area >= 12 and dominance >= .7 and not edge)
        result.append(row)
    return result
