#!/bin/bash

echo "***********************************************************************"
echo "                Starting the sphinx script"
echo "***********************************************************************"
echo "            Creating plugin API files and html pages .."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# members will document all modules
# undoc keeps modules without docstrings
# show-inheritance displays a list of base classes below the class signature

# Remove directory for api so that there are no obsolete files
rm -rf $DIR/source/generated/
rm -rf $DIR/build/

# sphinx-build [options] <sourcedir> <outputdir> [filenames]
# -a Write all output files. The default is to only write output files for new and changed source files. (This may not apply to all builders.)
# -E Don’t use a saved environment (the structure caching all cross-references), but rebuild it completely. The default is to only read and parse source files that are new or have changed since the last run.
# -b buildername, build pages of a certain file type
sphinx-build -a -E -b html $DIR/source/ $DIR/build/

echo "***********************************************************************"
echo "                          End of script"
echo "***********************************************************************"

