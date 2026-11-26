# Link with mold and compress debug sections. The development shell and the
# hosted check both use this string.
"-C link-arg=-fuse-ld=mold -C link-arg=-Wl,--compress-debug-sections=zlib"
