#!/bin/bash

# Array of search phrases
search_phrases=("dtype=np.float64")
# Array of replacement phrases
replacement_phrases=("dtype=np.float64")
 


# Check if the number of search phrases and replacement phrases match
if [ ${#search_phrases[@]} -ne ${#replacement_phrases[@]} ]; then
  echo "Error: The number of search phrases and replacement phrases does not match."
  exit 1
fi

# Execute substitution for each pair of search and replacement phrases
for (( i=0; i<${#search_phrases[@]}; i++ )); do
  search="${search_phrases[$i]}"
  replacement="${replacement_phrases[$i]}"
  command="grep -rlZ '$search' ./ | xargs -0 sed -i 's/$search/$replacement/g'"
  echo "Executing command: $command"
  eval $command
done
