import requests

# Path to the image file
image_path = "test_server.jpg"

# URL of the API endpoint
url = "https://neural-project.onrender.com/predictApi"

# Open the image file in binary mode
with open(image_path, "rb") as file:
    # Prepare the form data with the image file
    files = {"image": file}

    # Send the POST request
    response = requests.post(url, files=files)


# Check if the request was successful
if response.status_code == 200:
    print("Request was successful!")
    print("Response:", response.text)
else:
    print("Error:", response.status_code)
