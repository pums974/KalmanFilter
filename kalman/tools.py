#!/usr/bin/python2
# coding=utf-8
"""
    Tools
"""
import gc


def gc_clean():
    """
        Clean some memory
    :return:
    """
    n = gc.collect()
