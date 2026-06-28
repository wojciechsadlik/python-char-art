def print_gallery(char_arrs, columns=2):
    if not char_arrs:
        return

    joined_arr = []
    for i in range(0, len(char_arrs), columns):
        row_gallery = char_arrs[i: i + columns]

        item_width = len(row_gallery[0][0])
        h_sep_len = item_width * columns + 3 * columns + 3
        h_sep = '-' * h_sep_len

        joined_arr.append(h_sep)

        for lines in zip(*row_gallery):
            line_str = " | ".join("".join(line) for line in lines)
            joined_arr.append(f" | {line_str} | ")

    joined_arr.append('-' * h_sep_len)
    print('\n'.join(joined_arr))
