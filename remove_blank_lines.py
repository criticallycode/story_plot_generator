import pandas as pd

csv_in = "Horror_movies.csv"
csv_out = "Horror_movies_correct.csv"

#text_in = "Horror_plots_2.txt"
#text_out = "Horror_plots_2_correct.txt"

def csv_remove_blank(text_file, out_file):
    df = pd.read_csv(text_file, encoding = "ISO-8859-1")
    print (df.isnull().sum())
    modifiedDF = df.dropna()
    modifiedDF.to_csv(out_file, index=False)

def text_remove_blank(text_file, out_file):
    with open(text_file) as infile, open(out_file, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output

csv_remove_blank(csv_in, csv_out)
#text_remove_blank(text_in, text_out)