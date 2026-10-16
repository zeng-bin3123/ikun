#!/bin/bash
# Read the last 8 lines of each cmake log (the error part)
echo "OFF_LOG_LAST8:"
tail -8 /home/claude/ikun/cmake_off.log 2>/dev/null || echo "no file"

echo ""
echo "ON_LOG_LAST8:"
tail -8 /home/claude/ikun/cmake_on.log 2>/dev/null || echo "no file"

echo ""
echo "CMAKELISTS_LINES_40_89:"
sed -n '40,89p' /home/claude/ikun/core_cmake_copy.txt 2>/dev/null || echo "no file"
