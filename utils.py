def print_color(string, color_code, argument=""):
        # Black (Gray): 90m
        # Red: 91m
        # Green: 92m
        # Yellow: 93m
        # Blue: 94m
        # Magenta: 95m
        # Cyan: 96m
        # White: 97m
        print(f"\033[{color_code}m{string} {argument}\033[0m")