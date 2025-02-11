'''This file is for testing out various noise classes'''
from main import parse_noise_params, get_noisers, apply_noisers, record_noiser_artifacts, apply_noisers_compose
import os

# Character Level
# noise_specs = "character_level-lang=de,swap_theta=0.1"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# inputs = ["The quick brown fox jumps over the lazy dog", "जल में रहकर मगर से बैर"]

# for input in inputs:
#     print(f"Input: {input}")
#     noised = apply_noisers(input, noiser_classes, verbose=True)
#     print(f"Noised: {noised}")
#     print()

# Testing out lexical noiser with GlobalLexicalNoise
# lang = "fr"
# exp_key = f"lexical-lang={lang},theta_content_global=0.5,theta_func_global=0.8"
# read_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/datasets/{lang}/flores200-{lang}-en.txt"
# output_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/noiser_artifacts/test_{exp_key}"
# noise_specs = f"lexical-lang={lang},theta_content_global=0.5,theta_func_global=0.8,text_file=<{read_file}>,output_dir=<{output_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# with open(read_file, "r") as f:
#     inputs = f.readlines()[:20]

# output_dir = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/test_noised_outputs/{lang}/"
# os.makedirs(output_dir, exist_ok=True)
# noised_output_file = f"{output_dir}/{exp_key}.txt"
# out = open(noised_output_file, "w")

# for input in inputs:
#     print(f"Input: {input}", file = out)
#     noised = apply_noisers(input, noiser_classes, verbose=False)
#     print(f"Noised: {noised} \n\n", file = out)
# record_noiser_artifacts(noiser_classes)

# out.close()

# Phonological
# lang = "fr"
# exp_key = f"phonological-lang={lang},theta_phon=0.3"
# read_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/datasets/{lang}/arc_{lang}.txt"
# output_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/noiser_artifacts/test_{exp_key}"
# noise_specs = f"phonological-lang={lang},theta_phon=0.3,text_file=<{read_file}>,output_dir=<{output_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# with open(read_file, "r") as f:
#     inputs = f.readlines()[:20]

# output_dir = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/test_noised_outputs/{lang}/"
# os.makedirs(output_dir, exist_ok=True)
# noised_output_file = f"{output_dir}/{exp_key}.txt"
# out = open(noised_output_file, "w")

# for input in inputs:
#     print(f"Input: {input}", file = out)
#     noised = apply_noisers(input, noiser_classes, verbose=False)
#     print(f"Noised: {noised} \n\n", file = out)
#     record_noiser_artifacts(noiser_classes)

# out.close()


# Google Translate

# noise_specs = "gtrans-src=ru,tgt=be"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# inputs = ["Як правіла, брытанскія фунты будуць прымаць усюды на астравах.Стэнлі таксама часта прымае крэдытныя карты і долары ЗША."]

# for input in inputs:
#     print(f"Input: {input}")
#     noised = apply_noisers(input, noiser_classes, verbose=True)
#     print(f"Noised: {noised}")
#     print()

# # Morphological Noiser
# exp_key = "morph-lang=deu,theta_morph_global=0.5"
# read_file = "/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/datasets/de/arc_de.txt"
# output_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noiser_artifacts/test_{exp_key}"
# noise_specs = f"morph-lang=deu,theta_morph_global=0.5,text_file=<{read_file}>,output_dir=<{output_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# inputs = ["The quick brown fox jumps over the lazy dog"]

# # Some common German sentences:
# inputs = [
#     "Wie geht es dir?",  # How are you?
#     "Guten Morgen!",  # Good morning!
#     "Ich heiße Anna.",  # My name is Anna.
#     "Wo ist die Toilette?",  # Where is the toilet?
#     "Es tut mir leid.",  # I'm sorry.
#     "Was kostet das?",  # How much does it cost?
#     "Ich spreche ein wenig Deutsch.",  # I speak a little German.
#     "Danke schön!",  # Thank you very much!
#     "Woher kommst du?",  # Where are you from?
#     "Wie spät ist es?",  # What time is it?
#     "Auf Wiedersehen!",  # Goodbye!
#     "Ich verstehe nicht.",  # I don't understand.
#     "Wie heißt das auf Deutsch?",  # What is that called in German?
#     "Kann ich Ihnen helfen?",  # Can I help you?
#     "Ich liebe dich.",  # I love you.
#     "Wo wohnst du?",  # Where do you live?
#     "Bitte schön.",  # You're welcome.
#     "Wie alt bist du?",  # How old are you?
#     "Was machst du gerne?",  # What do you like to do?
#     "Entschuldigen Sie bitte.",  # Excuse me, please.
#     "Ich bin müde.",  # I'm tired.
#     "Gute Nacht!",  # Good night!
#     "Bis später!",  # See you later!
#     "Prost!",  # Cheers!
#     "Alles Gute zum Geburtstag!",  # Happy Birthday!
#     "Wie ist dein Name?",  # What is your name?
#     "Das ist lecker.",  # That is delicious.
#     "Ich komme aus Deutschland.",  # I am from Germany.
#     "Ich bin ein Tourist.",  # I am a tourist.
#     "Wo ist der Bahnhof?",  # Where is the train station?
#     "Ich habe mich verlaufen.",  # I am lost.
#     "Können Sie das bitte wiederholen?",  # Can you please repeat that?
#     "Wie viel Uhr ist es?",  # What time is it?
#     "Ich brauche Hilfe.",  # I need help.
#     "Wo ist das Hotel?",  # Where is the hotel?
#     "Es freut mich, Sie kennenzulernen.",  # Nice to meet you.
#     "Ich bin hier für eine Woche.",  # I am here for a week.
#     "Was ist das?",  # What is that?
#     "Wie war deine Reise?",  # How was your trip?
#     "Haben Sie Wasser?",  # Do you have water?
# ]

# for input in inputs:
#     print(f"Input: {input}")
#     noised = apply_noisers(input, noiser_classes, verbose=True)
#     print(f"Noised: {noised}")
#     print()
#     record_noiser_artifacts(noiser_classes)

# Morphological Noiser
# lang = "en"

# exp_key = f"morph-lang={lang},theta_morph_global=0.5"
# read_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/datasets/{lang}/arc_{lang}.txt"
# output_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/noiser_artifacts/test_{exp_key}"
# noise_specs = f"morph-lang={lang},theta_morph_global=0.5,text_file=<{read_file}>,output_dir=<{output_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# with open(read_file, "r") as f:
#     inputs = f.readlines()[:20]

# # Test output dir
# output_dir = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/test_noised_outputs/{lang}/"
# os.makedirs(output_dir, exist_ok=True)
# noised_output_file = f"{output_dir}/{exp_key}.txt"
# out = open(noised_output_file, "w")

# for input in inputs:
#     print(f"Input: {input}", file = out)
#     noised = apply_noisers(input, noiser_classes, verbose=False)
#     print(f"Noised: {noised} +\n\n", file = out)
#     record_noiser_artifacts(noiser_classes)

# out.close()

# Composite noise
## We'll apply lexical, phonological, and morphological noise

# lang = "de"
# exp_key = f"composite_deu_dan"
# read_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/datasets/{lang}/flores200-{lang}-en.txt"
# output_file = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/noiser_artifacts/test_{exp_key}"
# noise_specs = f"lexical-lang={lang},theta_content_global=0.5,theta_func_global=0.98,text_file=<{read_file}>;\
# morph-lang={lang},theta_morph_global=0.71,text_file=<{read_file}>;\
# phonological-lang={lang},theta_phon=0.1,text_file=<{read_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# read_file = "/export/b08/nbafna1/data/flores200_dataset/devtest/deu_Latn.devtest"
# with open(read_file, "r") as f:
#     inputs = f.readlines()[:20]

# output_dir = f"/export/b08/nbafna1/projects/llm-robustness-to-xlingual-noise/noisers/test_noised_outputs/{lang}/"
# os.makedirs(output_dir, exist_ok=True)
# noised_output_file = f"{output_dir}/{exp_key}.txt"
# out = open(noised_output_file, "w")

# for input in inputs:
#     print(f"Input: {input}", file = out)
#     # noised = apply_noisers(input, noiser_classes, verbose=False)
#     noised = apply_noisers_compose(input, noiser_classes, verbose=False)
#     print(f"Noised: {noised} \n\n", file = out)
# # record_noiser_artifacts(noiser_classes)

# out.close()

# lang = "hi"
# read_file = f"/export/b08/nbafna1/data/cvit-pib-v1.3/en-hi/train.hi"
# input_file = read_file
# with open(read_file, "r") as f:
#     inputs = f.readlines()


# noise_specs = f"lexical-lang={lang},theta_content_global=0.05,theta_func_global=0.5,text_file=<{read_file}>;morph-lang={lang},theta_morph_global=0.7,text_file=<{read_file}>;phonological-lang={lang},theta_phon=0.1,text_file=<{read_file}>"
# all_noise_params = parse_noise_params(noise_specs)
# print(f"Noise Parameters: {all_noise_params}")

# noiser_classes = get_noisers(all_noise_params)
# print(f"Noiser Classes: {noiser_classes}")

# inputs = ["कृषि मंत्री ने कहा कि छोटे ओैर सीमांत दूध उत्पादकों ने मिलकर 1998 से देश को विश्व का सबसे बड़ा दूध उत्पादक बनाने में योगदान दिया है ।"]
# for input in inputs:
#     # noised = apply_noisers(input, noiser_classes, verbose=False)
#     noised = apply_noisers_compose(input, noiser_classes, verbose=False)
#     print(noised)

lang = "hi"
read_file = f"/export/b08/nbafna1/data/cvit-pib-v1.3/en-hi/train.hi"
# input_file = read_file
# with open(read_file, "r") as f:
#     inputs = f.readlines()


noise_specs = f"lexical-lang={lang},theta_content_global=0.05,theta_func_global=0.5,text_file=<{read_file}>"
all_noise_params = parse_noise_params(noise_specs)
print(f"Noise Parameters: {all_noise_params}")

noiser_classes = get_noisers(all_noise_params)
print(f"Noiser Classes: {noiser_classes}")

inputs = ["कृषि मंत्री ने कहा कि छोटे ओैर सीमांत दूध उत्पादकों ने मिलकर 1998 से देश को विश्व का सबसे बड़ा दूध उत्पादक बनाने में योगदान दिया है ।"]
for input in inputs:
    noised = apply_noisers(input, noiser_classes, verbose=False)
    # noised = apply_noisers_compose(input, noiser_classes, verbose=False)
    print(noised)