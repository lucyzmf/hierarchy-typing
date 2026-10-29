import colorcet
from matplotlib.colors import LinearSegmentedColormap


def create_custom_cmap(rgb_value, transparencies: list):
    """create custom cmap with transparencies gradients"""
    color_list = [(*rgb_value[:3], a) for a in transparencies]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", color_list)
    return cmap


def custom_formatter(x, pos):
    # Check if x is exactly zero
    if x == 0:
        return "0"

    # Convert x to string with two decimal places
    formatted_str = f"{x:.2f}"

    # Remove leading zeros and unnecessary trailing zeros and decimal point
    formatted_str = formatted_str.lstrip("0").rstrip("0").rstrip(".")

    # Handle negative numbers formatting
    if formatted_str.startswith("-0"):
        formatted_str = "-" + formatted_str[2:]

    return formatted_str


# get colors
features = ["ContextEmbedding", "SpacyEmbedding", "SyllableEmbedding", "ButtonClass"]
cmap = colorcet.cm.rainbow_r

feat_color = {}

for f in range(len(features)):
    feat_color[features[f]] = cmap(f / (len(features) - 1))
