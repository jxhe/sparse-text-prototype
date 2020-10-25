#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-03-02 Junxian <He>
#
# Distributed under terms of the MIT license.

import sys

if __name__ == '__main__':
    save = set()
    for line in sys.stdin:
        key = line.rstrip()

        # for mscoco
        # key = '\t'.join(line.rstrip().split('\t')[1:])
        if key not in save:
            print(line.rstrip())
            save.update([key])

