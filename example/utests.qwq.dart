// ignore_for_file: avoid_print

import 'dart:io';

import 'package:llama_cpp_dart/llama_cpp_dart.dart';

String specs = """
{
  "unitTestScenario": {
    "scenarioName": "Example User Flow - Login and Post",
    "applicationUnderTest": "MyMobileApp",
    "platform": "Android",
    "version": "1.2.3",
    "devices": [
      {
        "deviceName": "Pixel 5",
        "osVersion": "Android 13",
        "screenResolution": "1080 x 2340"
      },
      {
        "deviceName": "Samsung Galaxy A53",
        "osVersion": "Android 12",
        "screenResolution": "1080 x 2400"
      }
    ],
    "steps": [
      {
        "stepNumber": 1,
        "description": "Launch the application.",
        "actionType": "appLaunch",
        "input": null,
        "expectedOutput": "App homescreen is displayed, showing the login form.",
        "validation": {
          "type": "elementVisible",
          "selector": "id:loginFormContainer",
          "message": "Verify the login form container is present on the screen."
        },
        "notes": "Ensure no initial error messages are shown. Network connectivity assumed."
      },
      {
        "stepNumber": 2,
        "description": "Enter valid username.",
        "actionType": "userInput",
        "input": {
          "fieldId": "usernameField",
          "value": "testuser"
        },
        "expectedOutput": "Username field populated with 'testuser'.",
        "validation": {
          "type": "textFieldValue",
          "fieldId": "usernameField",
          "expectedValue": "testuser"
        },
        "notes": "Username is pre-existing in the test database. Consider different username scenarios (e.g., empty, too long, special characters) in other tests."
      },
      {
        "stepNumber": 3,
        "description": "Enter valid password.",
        "actionType": "userInput",
        "input": {
          "fieldId": "passwordField",
          "value": "password123"
        },
        "expectedOutput": "Password field populated with 'password123'.",
        "validation": {
          "type": "textFieldValue",
          "fieldId": "passwordField",
          "expectedValue": "password123",
          "secure": true
        },
        "notes": "Password field is masked for security reasons."
      },
      {
        "stepNumber": 4,
        "description": "Tap the Login button.",
        "actionType": "buttonTap",
        "input": {
          "buttonId": "loginButton"
        },
        "expectedOutput": "Successful login. Redirect to the main app feed.",
        "validation": {
          "type": "pageTransition",
          "fromPage": "loginPage",
          "toPage": "feedPage",
          "message": "Verify redirection to the feed page."
        },
        "notes": "Check for any error messages related to network issues."
      },
      {
        "stepNumber": 5,
        "description": "Tap the 'New Post' button.",
        "actionType": "buttonTap",
        "input": {
          "buttonId": "newPostButton"
        },
        "expectedOutput": "Post creation form displayed.",
        "validation": {
          "type": "elementVisible",
          "selector": "id:postFormContainer",
          "message": "Verify post creation form appears."
        },
        "notes": "Confirm that the transition to the new post page is smooth."
      },
      {
        "stepNumber": 6,
        "description": "Enter a short text post.",
        "actionType": "userInput",
        "input": {
          "fieldId": "postText",
          "value": "This is a test post!"
        },
        "expectedOutput": "Text post field populated with 'This is a test post!'.",
        "validation": {
          "type": "textFieldValue",
          "fieldId": "postText",
          "expectedValue": "This is a test post!"
        },
        "notes": "Consider tests for post length limitations, special character handling."
      },
      {
        "stepNumber": 7,
        "description": "Tap the 'Post' button.",
        "actionType": "buttonTap",
        "input": {
          "buttonId": "postButton"
        },
        "expectedOutput": "Post is published. Redirect to feed and the new post appears.",
        "validation": {
          "type": "pageTransition",
          "fromPage": "newPostPage",
          "toPage": "feedPage",
          "message": "Verify redirect back to the feed."
        },
        "notes": "Validate the newly created post appears correctly on the feed (e.g., timestamp, author)."
      },
       {
        "stepNumber": 8,
        "description": "Verify a post appears with content.",
        "actionType": "elementPresent",
        "input": {
            "selector": "xpath://android.widget.TextView[@resource-id='postId' and text()='This is a test post!']"
        },
        "expectedOutput": "Text appears in a TextView.",
        "validation": {
           "type": "elementPresent",
           "selector": "xpath://android.widget.TextView[@resource-id='postId' and text()='This is a test post!']"
        }
       }
    ],
    "cleanup": "Delete the created test post from the database after testing is complete.",
    "preConditions": [
      "Application installed and running.",
      "Test database initialized and populated with a user named 'testuser'.",
      "Device has an active network connection."
    ],
    "postConditions": [
      "Application state reverted to the pre-test state.",
      "Database cleaned up (if necessary)."
    ]
  }
}
""";

void main() async {
  try {
    ContextParams contextParams = ContextParams();
    contextParams.nPredict = -1;
    contextParams.nCtx = 512 * 3;
    contextParams.nBatch = 512 * 3;

    final samplerParams = SamplerParams();
    samplerParams.temp = 0.6;
    samplerParams.minP = 0;
    samplerParams.topK = 20;
    samplerParams.topP = 0.95;
    // samplerParams.penaltyRepeat = 1.1;

    Llama.libraryPath = "bin/MAC_ARM64/libllama.dylib";
    String modelPath = "/Users/adel/Workspace/gguf/qwq-32b-q4_k_m.gguf";
    Llama llama =
        Llama(modelPath, ModelParams(), contextParams, samplerParams, true);

    llama.setPrompt("""
based on example of unit test instructions
$specs

please generate a unit test to: fuzz buzz
""");
    while (true) {
      var (token, done) = llama.getNext();
      stdout.write(token);
      if (done) break;
    }
    stdout.write("\n");

    llama.dispose();
  } catch (e) {
    print("Error: ${e.toString()}");
  }
}
