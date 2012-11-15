#!/usr/bin/bash

if [ -z $1 ] ; then
    echo "USAGE: symlink_images.sh runs_dir"
    exit
fi
base_dir=$1

# find "${base_dir}" -name 'gibbs_init_state.png' | perl -ne '
find "${base_dir}" -regex '.*\(ari\|score\|clusters\|test\|alpha\|beta\).*pdf' | perl -ne '
$orig=$_;
$orig=~s/\s+$//;
@orig_parts = split(/\//, $orig);
$num_parts = @orig_parts;
$new = $orig_parts[-2] . "_" . $orig_parts[-1];
print "ln -s $orig $new\n"' \
| bash