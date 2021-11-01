#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path

path = os.getcwd()
for root, dirs, files in os.walk(path):
    for name in files:
        if (name.endswith(".h") or name.endswith(".c")):
            localpath = root + '/' + name
            os.system("clang-format -i %s" % (localpath))
