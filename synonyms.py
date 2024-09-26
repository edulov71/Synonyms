# Use your own Google Books API key

"""
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
"""
import nltk
from nltk.corpus import wordnet
import requests
import json
import numpy as np

# Ensure you have downloaded WordNet
nltk.download('wordnet')
# Use nltk's WordNet to get synonyms of the given word
### Input: string containing one word, 3 character length at least.
### Output: list of strings representing one-word synonyms or empty list when no synonyms were found
# or the word is unique, or probably mistyped
# Making the set/list of synonyms manually, we offer space for additional operations.
# For example, calling wordnet.synsets('dog', pos=wordnet.VERB) will return word "chase"
def get_synonyms_by_lemmas(word: str) -> list[str]:
    # Get all synsets (sets of synonymous words) for the given word
    word = word.lower().lstrip().rstrip()
    if len(word)>2:
        synsets = wordnet.synsets(word)
        synonyms = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                # Add the lemmas (synonyms) to the set
                wrd = lemma.name().lower()
                # We do not need composite synonyms
                if "_" not in wrd:
                    synonyms.add(wrd)
        # Remove duplicates and initial word; and return as a list
        return list(synonyms - {word})
    else:
        return []

# These two functions are almost the same and wordnet.synonyms() will produce the same result, but we truncate
# "wider" synonyms from our scope
def get_synonyms(word: str) -> list[str]:
    # Get all synsets (sets of synonymous words) for the given word
    word = word.lower().lstrip().rstrip()
    if len(word)>2:
        # since the main function synonyms() returns list of the lists for possible candidates,
        # we will use the "main" i.e. the unique elements from the first sublist only, removing composite words too
        return [wrd for wrd in set(wordnet.synonyms(word)[0]) if "_" not in wrd]
    else:
        return []

# To get the synonyms arranged in accordance to their appearence frequences in real texts,
# we are using Google's Book API. Thus a user should provide a valid API key.
# Here we try to push a but and make several attempts when connection fails.
# We do not look for a specific frequency and calculate it's mean value over all years reported by API
### Input: the word to be processed
### Output: a float/real number in terms of internal Google's metrics. Zero will be returned if no such word exists or
# some connection/API failure appeared
def get_synonym_frequency(word : str,googleAPI_key : str ="") -> float:
    """
    Fetch synonym frequencies from Google N-Gram Viewer API.
    """
    # corpus data may start at 1900
    # corpus en. en-US, en-GB, en-2012 (from 1970), fr
    url = f"https://books.google.com/ngrams/json?content={word}&year_start=1800&year_end=2024&corpus=ru&smoothing=3&key={googleAPI_key}"
    
    timeout_attempts = 3
    while timeout_attempts>0:
        try: # up to 3 seconds for establishing connection and sending rrequest
             # up to 8 seconds if server is slow in generating result or/and connection is slow while the response size is lengthy
            resp = requests.get(url, timeout=(3, 8))
        except requests.exceptions.Timeout:
            timeout_attempts -= 1
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            break
        else:
            # We got response, leaving aux loop
            break
    if resp.ok:
        try:
            data = json.loads(resp.content)
        except:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed')
            return 0.0
        else:
        # Extract the frequency from the first result if it exists or zero otherwise
            frequency = 0.0
            if len(data)>0:
                res = np.array(data[0]['timeseries'])
                # We have to cautious here, since we are requesting a wide timeline data, thus for newer words this
                # sequence may contain zeros and standard average will compute a logically invalid value.
                # Thus we have to do it manually
                frequency =np.true_divide(res.sum(0), (res != 0).sum(0))
        return frequency
    else:
        print(f"Failed to fetch frequency for '{word}'")
        return 0.0

# Getting a <dumb> list of synonyms is not sufficient in most cases. Normally we want to get a list
# where synonyms will be ordered from most frequent towards the least frequent.
# Also in some cases we want to measure their frequence relatively, using the most frequent synonym (not the initial word itself)
# as a baseline. In this case user must provide a relative threshold margin (0,1) as a reference.
# Setting this threshold close to one will produce a list of the really popular words or the "best" synonym only for 1.0
# The value like 0.25 means that we want to exclude all synonyms which are more then 1/0.25=4 times less frequent
### Input: string containing one word, 3 character length at least and "domination rate" optionally if we want are not interested
# in less frequent (unpopular) words as synonyms.
# !!! Be careful. It is not recommended to set this threshold when we deal with terminology
# !!! For example, using "test" as an input, the extended list of synonyms is:
# !!! ['essay', 'exam', 'examination', 'examine', 'prove', 'quiz', 'run', 'screen', 'trial', 'try', 'tryout']
# !!! while setting 0.25 threshold we will get only
# !!! [('trial', 1.0), ('run', 0.95757), ('examination', 0.85081), ('prove', 0.77545), ('try', 0.56084), ('examine', 0.35763)]
### Output: two empty lists when no synonyms were found, but in normal case these lists are:
# >> list of tuples where the first element is a synonym, second element is its relative frequence rounded to 5 significant digits in mantissa.
# Please do understand that those values are somehow abstract and keeping a whole bunch of digits is meaningless
# >> ordinary list of strings which represents the whole list of one-word synonyms sorted in lexicographic order for the sake of generality
def get_most_frequent_synonyms(word : str,domination_rate = None|float) -> tuple[list[tuple[str,float]], list[str]]:
    synonyms = get_synonyms(word)
    if len(synonyms)>0:
        frequency_dict = {synonym: get_synonym_frequency(synonym) for synonym in synonyms}
        sorted_synonyms = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        # Ignore words w/o valid frequency information. Either a non-existing word or no information received
        # making N-gram request
        base_value = sorted_synonyms[0][1]
        if domination_rate is None or domination_rate<0 or domination_rate>1:
            return [(synonym,np.round(freq/base_value,5)) for synonym, freq in sorted_synonyms if freq>0],sorted(synonyms)
        else:
            threshold = domination_rate*base_value
            return [(synonym,np.round(freq/base_value,5)) for synonym, freq in sorted_synonyms if freq>=threshold],sorted(synonyms)
    else:
        return [],[]

# Works similar to the _most_ function w/o forced threshholding, but will only report synonyms, more popular than "base word". It means that in many cases
# a user will get a list of possible synonyms as a second returned list, while the first list will be empty
# When the first list is not empy, please recall that the ratio (second value in the tuple) is greater then 1, and it could
# be notable greater, when the input word is not popular. Also one should understand that keeping lots of digits in mantissa for
# this particular task is meaningless, thus ratio is rounded to two digits only
# For example, asking about "tryout", we will get something like that: [('test', 824.62), ('trial', 141.58)]
def get_more_frequent_synonyms(word : str) -> tuple[list[tuple[str,float]], list[str]]:
    synonyms = get_synonyms(word)
    if len(synonyms)>0:
        frequency_dict = {synonym: get_synonym_frequency(synonym) for synonym in [word]+synonyms}
        base_value = frequency_dict[word]
        sorted_synonyms = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        # Ignore words w/o valid frequency information. Either a non-existing word or no information received
        # making N-gram request
        # to exclude <word> from the list of results frequency has to be strictly greater then <base_value>
        return [(synonym,np.round(freq/base_value,2)) for synonym, freq in sorted_synonyms if freq>base_value],sorted(synonyms)
    else:
        return [],[]

# Example usage
word = "tryout"
synonyms,complete_list = get_most_frequent_synonyms(word,domination_rate=.25)
print(f"Most frequent synonyms for '{word}' are: {synonyms}")
print(f"Complete list of synonyms is: {complete_list}")
synonyms_m,_ = get_more_frequent_synonyms(word)
print(f"More frequent then '{word}' synonyms are: {synonyms_m}")
