from openai import OpenAI

client = OpenAI() #Open AI client auto reader

# Test function to verify OpenAI integration
def main():
    response = client.responses.create(
        model = "gpt-5-nano",
        input = "Say hello to Michael from Nigeria who dreams of studying in the UK.",
        max_output_tokens=50
    )
    print(response.output_text)
    print("Test file for OpenAI integration.")
    # Additional test assertions can be added here
if __name__ == "__main__":
    main()
