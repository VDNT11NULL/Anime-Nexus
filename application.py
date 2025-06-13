# from flask import Flask, request, render_template, jsonify
# from pipelines.prediction_pipeline import hybrid_rec_sys

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     recommendations = None
#     if request.method == 'POST':
#         try:
#             user_id = int(request.form["user_id"])
#             recommendations = hybrid_rec_sys(user_id)
#         except Exception as e:
#             print(f'Error occured while recommending... : {e}')
#     return render_template('index.html', recommendations=recommendations)

# if __name__=='__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, render_template, jsonify
from pipelines.prediction_pipeline import hybrid_rec_sys

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            user_id = int(request.form.get("user_id"))
            if user_id <= 0:
                raise ValueError("User ID must be a positive integer")
            recommendations = hybrid_rec_sys(user_id, final_num2_rec=5)  # Limit to 5 recommendations
            return jsonify({"success": True, "recommendations": recommendations})
        except ValueError as ve:
            return jsonify({"success": False, "error": f"Invalid input: {str(ve)}"})
        except Exception as e:
            return jsonify({"success": False, "error": f"An error occurred: {str(e)}"})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)