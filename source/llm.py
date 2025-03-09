from openai import OpenAI

def gen_llm_description(client, title, description, llm):
    """
    Generates a description for a youtube video based on title and description provided
    """
    system_prompt = (
        "You are an AI assistant specialized in generating informative YouTube video descriptions. Avoid links, hashtags, or any promotional content."
        "Your task is to provide a professional and engaging description for a YouTube video based on the title and provided description."
    )

    user_prompt = (
        f"Generate a professional and engaging YouTube video description for the following video:\n\n"
        f"**Title:** {title}\n"
        f"**Provided Description:** {description}\n\n"
        f"Ensure the output is direct, informative, and compelling. No additional commentary—just the description."
    )
    
    answer = ""
    while answer == "":
        completion = client.chat.completions.create(
            temperature=0.1,
            max_tokens=150,
            model=llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
        )
        answer = completion.choices[0].message.content.strip()
    return answer

def gen_llm_report(client, transcript, llm):
    """Generate a report in markdown format using OpenAI's client."""
    system_prompt = "You are an AI assistant that generates structured markdown reports from transcripts. The report should be well-organized with sections and subsections as needed."
    user_prompt = f"Generate a well-structured markdown report from the following transcript:\n\n{transcript}\n\nEnsure proper sectioning and clear formatting."
    answer = ""
    while answer == "":
        completion = client.chat.completions.create(
            model=llm,
            temperature=0.1,
            max_tokens=5000,
            messages=[
                {"role": "system", "content": system_prompt},                
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = completion.choices[0].message.content.strip()

    return answer