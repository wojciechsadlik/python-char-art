from itertools import zip_longest

def print_gallery(char_arrs, columns=2):
    if not char_arrs:
        return

    joined_arr = []
    for i in range(0, len(char_arrs), columns):
        row_gallery = char_arrs[i : i + columns]

        # Calculate character width for each item so missing lines can be padded
        item_widths = [len(img[0]) if img else 0 for img in row_gallery]
        
        items_width = sum(item_widths)
        h_sep_len = items_width + 3 * len(row_gallery) + 1
        h_sep = '-' * h_sep_len

        joined_arr.append(h_sep)

        for lines in zip_longest(*row_gallery, fillvalue=None):
            formatted_lines = []
            for img_idx, line in enumerate(lines):
                if line is not None:
                    formatted_lines.append("".join(line))
                else:
                    formatted_lines.append(" " * item_widths[img_idx])
            
            line_str = " | ".join(formatted_lines)
            joined_arr.append(f"| {line_str} |")

        joined_arr.append(h_sep)

    print('\n'.join(joined_arr))
