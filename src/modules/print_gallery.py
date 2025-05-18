def print_char_arrs(char_arrs, columns=2):
    gallery = []
    for i in range(len(char_arrs)):
        if (i % columns == 0):
            gallery.append([])
        gallery[-1].append(char_arrs[i])

    h_sep_len = len(char_arrs[0][0]) * columns + 3 * columns + 3
    joined_arr = []
    for row in gallery:
        h_sep = ['-' for _ in range(h_sep_len)]
        joined_arr.append(''.join(h_sep))

        for y in range(len(row[0])):
            new_row = ' | '
            for char_arr in row:
                new_row += ''.join(char_arr[y]) + ' | '
            joined_arr.append(new_row)

    h_sep = ['-' for _ in range(h_sep_len)]
    joined_arr.append(''.join(h_sep))

    print('\n'.join(joined_arr))
