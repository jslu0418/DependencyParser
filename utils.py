def format_string(s):
    if len(s) < 8:
        s = s + '\t'
    return s


def print_title():
    print('-' * 79)
    print('{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format('Idx','Token','POSTag','Head(o)','Label(o)','Head(p)','Label(p)','match_h','match_l'))
    print('-' * 79)
