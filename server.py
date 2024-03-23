from flask import Flask, jsonify, request


app=Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return "Summary Generator"

@app.route('/summary_generate', methods=['GET', 'POST'])
def summary_generator():
    encode_url=unquote(unquote(request.args.get('url')))
    if not encode_url:
        return jsonify({'error':'URL is required'}), 400
    text=extract_data_website(encode_url)
    #text_chunks=split_text_chunks(text)
    #print(len(text_chunks))
    summary=split_text_chunks_and_summary_generator(text)
    print("Here is the Complete Summary", summary)
    response= {
        'submitted_url': encode_url,
        'summary': summary
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)
