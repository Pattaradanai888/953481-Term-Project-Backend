from app import create_app
import traceback

try:
    app = create_app()

    if __name__ == '__main__':
        app.run(debug=True)
except Exception as e:
    print("Error starting the Flask app:")
    print(str(e))
    print(traceback.format_exc())