#!/usr/bin/env python
"""
Entry point wrapper for backwards compatibility.

Usage:
    python run.py

This is a thin wrapper that calls the main function from the hmm_var package.
For new code, prefer using:
    python -m hmm_var.main
    or
    hmm-var (after pip install -e .)
"""

from hmm_var.main import main

if __name__ == "__main__":
    main()
