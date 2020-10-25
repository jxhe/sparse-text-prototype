"""
This script collect all results in a directory
"""

import os
import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_line(line):
    line_s = line.strip().split(" | ")
    result = {}
    for entry in line_s:
        if len(entry.split()) == 2:
            k, v = entry.split()
            if is_number(v):
                result[k] = v

    return result


parser = argparse.ArgumentParser(description="collect results script")

parser.add_argument('--outdir', type=str, help="an high level experiment dir")

args = parser.parse_args()

best_results = []
best_nll = 1000
best_ppl = 1000

with open(os.path.join(args.outdir, "summary.txt"), "w") as fout:
    for root, subdirs, files in os.walk(args.outdir):
        valid_result = {}
        best_nll = best_ppl = 1000
        best_line_str = ''
        last_line_str = ''

        best_line_dict = {}
        last_line_dict = {}
        for file in files:
            if file == "stdout.log":
                print("processing {}".format(os.path.join(root, file)))
                fin = open(os.path.join(root, file))
                for line in fin:
                    if "valid on" in line and 'ppl_iw' not in line:
                        valid_result = parse_line(line)
                        if len(valid_result) > 0 and float(valid_result['loss']) > 0:
                            ppl = float(valid_result["ppl"])
                            if ppl < best_ppl:
                                best_ppl = ppl
                                best_line_str = line.rstrip()
                                best_line_dict = valid_result

                            last_line_str = line.rstrip()
                            last_line_dict = valid_result

                fin.close()
                break

        if len(valid_result) > 0:
            try:
                fout.write("{}\n".format(os.path.abspath(root)))
                fout.write("valid best loss: {}, best ppl: {}\n".format(
                    best_line_dict["loss"], best_line_dict["ppl"]))
                fout.write("valid last loss: {}, last ppl: {}\n".format(
                    last_line_dict["loss"], last_line_dict["ppl"]))
                fout.write("{}\n".format(best_line_str))
                fout.write("\n-----------------------------------\n\n")
            except KeyError:
                pass