#!/usr/bin/env python
"Merges same shape latex tabulars."

from optparse import OptionParser

def merge_table(file_list):
    ref_file = open(file_list[0])
    files = [open(file_list[i]) for i in range(1,len(file_list))]

    pass


def main():
    usage = "%prog -r tex_file_1 -r tex_file_2 ... -r tex_file_n"

    parser = OptionParser(usage = usage)
    parser.add_option("-r", dest = "file_list", type = "string",
                      action = "append", help = "input latex files") 
    (options, args) = parser.parse_args()

    if not options.file_list:
        parser.print_help()
        exit()
	
    merge_table(options.file)

if __name__ == '__main__':
    main()
