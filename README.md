# Synonyms
Using NLTK and Google Book's API to obtain lists of synonyms as-is and by their frequences

code example:

word = "tryout"
synonyms,complete_list = get_most_frequent_synonyms(word,domination_rate=.25)
print(f"Most frequent synonyms for '{word}' are: {synonyms}")
print(f"Complete list of synonyms is: {complete_list}")
synonyms_m,_ = get_more_frequent_synonyms(word)
print(f"More frequent then '{word}' synonyms are: {synonyms_m}")

should produce an output similar to this one:

[nltk_data] Downloading package wordnet to /home/edh/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
Most frequent synonyms for 'tryout' are: [('test', 1.0)]
Complete list of synonyms is: ['test', 'trial']
More frequent then 'tryout' synonyms are: [('test', 824.62218), ('trial', 141.58117)]
