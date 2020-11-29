data_dir=$1
python encode_doc_words_norm.py ${data_dir}
#python format_data_for_gnn.py ${data_dir}

cd ${data_dir}
cat doc_word.edges word_word.edges > data.edges
