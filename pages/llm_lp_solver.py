import streamlit as st
from google import genai
# from google.genai import types # Import types if needed for advanced config
import os
import re # Import the regular expression module
from dotenv import load_dotenv
# --- Required for capturing exec() output ---
import io
from contextlib import redirect_stdout
# --- Ensure numpy and scipy are available for exec() ---
# You might need to add these to your requirements.txt: pip install scipy numpy
import numpy
from scipy.optimize import linprog

# Load environment variables from .env file
load_dotenv()

# --- Add Scipy import for potential use later (optional, but good practice if you plan to run the code) ---
# Note: This is not strictly needed just to *generate* the code,
# but useful if you ever want to execute it within Streamlit.
# You might need to add scipy to your requirements.txt: pip install scipy
# from scipy.optimize import linprog

st.set_page_config(page_title="LLM LP Solver", page_icon="🧠")

# Initialize client and model variables to None initially
client = None
api_key_configured = False

# --- Helper function to extract LaTeX part ---
# Updated Cleanup: Iteratively remove common artifacts from start/end. Wrap in align*.
def extract_latex_formulation(text):
    """
    Extracts the concise mathematical formulation section (expected to be in LaTeX)
    from the LLM response. Looks for marker heading, captures until end of string.
    Cleans up potential leading/trailing markdown/code fences/text artifacts iteratively.
    Wraps the result in an align* environment for st.latex.
    """
    match = re.search(r"Concise LaTeX Formulation:?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        formulation = match.group(1).strip() # Group 1 is everything after the heading

        # --- Refined Cleanup: Iteratively clean start and end ---
        prev_formulation = ""
        # Loop until no more artifacts are removed in a pass
        while formulation != prev_formulation:
            prev_formulation = formulation

            # 1. Remove code fences (``` or ```latex) and surrounding whitespace
            formulation = re.sub(r"^\s*```(?:latex)?\s*", "", formulation)
            formulation = re.sub(r"\s*```\s*$", "", formulation)

            # 2. Remove bold markers (**) and surrounding whitespace
            formulation = re.sub(r"^\s*\*\*+\s*", "", formulation)
            formulation = re.sub(r"\s*\*\*+\s*$", "", formulation)

            # 3. Remove literal "latex" word (case-insensitive) if it's at the very start, plus surrounding whitespace/punctuation
            # Use regex to handle potential leading punctuation before "latex"
            formulation = re.sub(r"^\s*[^a-zA-Z0-9\\\{\(\s]*latex\s*", "", formulation, flags=re.IGNORECASE)

            # 4. Remove common leading/trailing punctuation/whitespace (quotes, colons, periods etc.)
            # Be careful not to remove essential math symbols or characters
            # Corrected regex character sets to avoid syntax errors
            formulation = re.sub(r"^\s*['\"“”:.]+\s*", "", formulation) # Leading specific punctuation
            formulation = re.sub(r"\s*['\"“”:.]+\s*$", "", formulation) # Trailing specific punctuation

            # Final strip for the iteration
            formulation = formulation.strip()
        # --- End Refined Cleanup ---


        # Remove any existing \begin{...} \end{...} blocks LLM might have added
        # This is important if the LLM correctly followed the prompt AND added fences
        formulation = re.sub(r"\\begin\{aligned\*?\}", "", formulation, flags=re.DOTALL).strip()
        formulation = re.sub(r"\\end\{aligned\*?\}", "", formulation, flags=re.DOTALL).strip()

        # Ensure we captured something meaningful
        if formulation:
             # Wrap the extracted content in align* environment
             return rf"\begin{{align*}}{formulation}\end{{align*}}"
    # If extraction failed or resulted in empty string, return None
    return None


# Try to get the API key and initialize the client
try:
    # Get the API key from the environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Raise an error if the key is not found or is empty
        raise ValueError("GEMINI_API_KEY not found in .env file or environment.")

    # Initialize the client using the API key
    client = genai.Client(api_key=api_key)

    # Flag to indicate if the API key is available and client is configured
    api_key_configured = True

except (ValueError, KeyError) as e:
    # Handle the case where the API key is not set in the environment
    st.error(f"🛑 Error initializing Google AI Client: {e}")
    st.error("Please ensure your GEMINI_API_KEY is set in a `.env` file in your project root.")
    st.code("""
# .env
GEMINI_API_KEY="YOUR_API_KEY"
    """, language="text")
    api_key_configured = False
except Exception as e:
    # Catch other potential initialization errors (like authentication issues)
    st.error(f"Failed to initialize Google AI Client: {e}")
    api_key_configured = False


st.markdown("# LLM LP Solver")
st.sidebar.header("LLM LP Solver Options")
st.write(
    """
    Enter a description of an optimization problem below.
    The AI will attempt to formulate it mathematically.
    """
)

# Add a text input field
# Disable input if the client wasn't successfully configured
prompt = st.text_input("Enter your problem description or prompt here:", disabled=not api_key_configured)

if prompt and api_key_configured:
    st.write(f"**You entered:**\n\n{prompt}") # Show the original input

    # *** Enhanced Prompt using Prompt Engineering Techniques ***
    llm_prompt = f"""
            **Role:** You are an expert assistant specializing in Mathematical Optimization and LaTeX formatting. Your primary goal is to accurately convert problem descriptions into Linear Programming formulations.

            **Task:** Convert the user's problem description below into a standard Linear Programming (LP) formulation. Provide both a detailed explanation and a concise, renderable LaTeX block.

            **Input Problem Description:**
            {prompt}

            **Output Requirements:**
            Produce exactly two distinct sections, separated by '---'.

            **SECTION 1: Formulation Explanation**
            *   Use clear headings (e.g., **Decision Variables**, **Objective Function**, **Constraints**).
            *   Explain the

                Formulation Explanation:
                [Provide the detailed explanation here using plain text or markdown]

                ---
                Concise LaTeX Formulation:
                [Provide *only* the final mathematical formulation here, written directly in LaTeX syntax. Use commands like \\max, \\min, \\text{{Subject to:}}, \\le, \\ge, x_1, x_2, etc. Please use alignment characters (&) for proper alignment and wrap the entire formulation (objective and constraints) in a LaTeX 'aligned' environment (\\begin{{aligned}} ... \\end{{aligned}}) for proper display.]
                """

    try:
        # Add a spinner while waiting for the API response
        with st.spinner("🧠 Thinking... Please wait."):
            # Generate content using the client
            response = client.models.generate_content(
                model='gemini-2.0-flash', # Or your desired model
                contents=llm_prompt,
                # Consider adding safety settings if needed
            )

        full_response_text = response.text

        # --- Add temporary debug output ---
        # st.subheader("Debug: Full LLM Response")
        # st.text_area("Full Response", full_response_text, height=200)
        # --- End temporary debug output ---

        # --- Display Full Explanation FIRST ---
        st.divider() # Add a visual separator
        st.subheader("📝 Full AI Explanation:")
        # Display the full response from the LLM
        st.markdown(full_response_text) # Use markdown to render potential formatting


        # --- Display Concise LaTeX Formulation ---
        st.divider() # Add another separator
        st.subheader("📊 Concise Formulation (LaTeX):")
        latex_formulation = extract_latex_formulation(full_response_text) # Extract, clean, and wrap

        # --- Remove or comment out the DEBUG section ---
        # st.subheader("Debug Info")
        # if latex_formulation:
        #     st.text("Extracted & Wrapped LaTeX String (for st.latex):")
        #     st.text_area("Raw Extracted & Wrapped LaTeX", latex_formulation, height=150)
        # else:
        #     st.text("Extraction returned None.")
        # st.divider()
        # --- END DEBUGGING ---


        if latex_formulation:
            try:
                st.latex(latex_formulation)
            except Exception as latex_error:
                st.error(f"Failed to render LaTeX. Extracted text was:\n```latex\n{latex_formulation}\n```\nError: {latex_error}")
                st.warning("The LLM might not have formatted the 'Concise LaTeX Formulation' section exactly as expected, or the LaTeX might contain errors.")
            
            # --- NEW: Section to generate Scipy code ---
            st.divider()
            st.subheader("🐍 Python Scipy Code & Solution:") # Updated subheader

            # Create the prompt for code generation
            code_gen_prompt = f"""
            **Role:** You are an expert programmer specializing in Python and the SciPy library for scientific computing, particularly `scipy.optimize.linprog`.

            **Task:** Convert the following Linear Programming problem formulation, provided in LaTeX format, into a Python script that uses `scipy.optimize.linprog` to solve it.

            **Input LP Formulation (LaTeX):**
            ```latex
            {latex_formulation}
            ```

            **Output Requirements:**
            *   Produce a complete, runnable Python code block.
            *   Include necessary imports (`numpy`, `scipy.optimize.linprog`).
            *   Correctly identify the objective function coefficients (`c`). Remember that `linprog` minimizes, so if the problem is maximization, negate the coefficients.
            *   Correctly identify the inequality constraint matrix (`A_ub`) and vector (`b_ub`). Convert any `>=` constraints to `<=` by multiplying both sides by -1.
            *   Correctly identify the equality constraint matrix (`A_eq`) and vector (`b_eq`).
            *   Define variable bounds (`bounds`), assuming non-negativity unless otherwise specified in the LaTeX.
            *   Include the call to `scipy.optimize.linprog` with the extracted parameters.
            *   Print the optimization result (e.g., print(res) or print specific attributes like res.x and res.fun). **Crucially, ensure the script *prints* the final result 
            object or its key attributes so they can be captured.** Structure the output (prints) in a way so that a LLM can easily understand and extract the solution.
            *   Wrap the final code in a single Python code block (```python ... ```). Do not include any explanatory text outside the code block.
            """

            try:
                # Add a spinner for the second API call
                with st.spinner("🐍 Generating and Executing Scipy code..."): # Updated spinner text
                    # Generate content using the client
                    code_response = client.models.generate_content(
                        model='gemini-2.0-flash', # Or your preferred model
                        contents=code_gen_prompt,
                        # safety_settings=... # Optional safety settings
                    )

                python_code = code_response.text

                # --- Clean up the response to extract only the Python code block ---
                extracted_code = None # Initialize extracted_code
                code_match = re.search(r"```python\s*(.*?)\s*```", python_code, re.DOTALL)
                if code_match:
                    extracted_code = code_match.group(1).strip()
                    st.code(extracted_code, language="python") # Display the generated code
                else:
                    # Fallback if the LLM didn't use the exact ```python fence
                    st.warning("Could not automatically extract the Python code block from the response. Displaying the full response:")
                    st.text_area("Full Code Generation Response", python_code, height=300)
                    # Attempt to use the full response if extraction failed but it looks like code
                    if "scipy.optimize.linprog" in python_code:
                         extracted_code = python_code # Try executing the full response

                # --- Execute the extracted code and capture output ---
                if extracted_code:
                    st.markdown("---") # Separator before execution results
                    st.subheader("⚙️ Execution Output:")

                    # Add a warning about exec()
                    st.warning("⚠️ **Security Note:** Executing LLM-generated code. Ensure the prompt is not malicious.", icon="⚠️")

                    output_capture = io.StringIO()
                    solution_output = "" # Initialize solution output string
                    try:
                        # Make numpy and linprog available to the executed code
                        exec_globals = {'numpy': numpy, 'linprog': linprog}
                        with redirect_stdout(output_capture):
                            exec(extracted_code, exec_globals)

                        solution_output = output_capture.getvalue() # Capture the output

                        if solution_output:
                             st.text("Solution Printed by the Script:")
                             st.code(solution_output, language="text")

                             # --- NEW: Section for Final Interpretation ---
                             st.divider()
                             st.subheader("💡 Interpretation of Results:")

                             # Create the prompt for interpretation
                             interpretation_prompt = f"""
                             **Role:** You are an expert assistant skilled at interpreting optimization results and explaining them clearly in the context of the original problem.

                             **Task:** Explain the provided optimization solution in plain language, relating it back to the user's original problem description.

                             **Original Problem Description:**
                             {prompt}

                             **Optimization Script Output (Solution):**
                             ```
                             {solution_output}
                             ```

                             **Output Requirements:**
                             *   Clearly state the optimal values of the decision variables found (if available in the output).
                             *   State the optimal objective function value (if available in the output).
                             *   Explain what these values mean *in the context of the original problem description* (e.g., "This means you should produce 5 units of product A and 10 units of product B to maximize profit...").
                             *   Focus on providing a clear, natural language explanation based *only* on the original problem and the provided solution data.
                             *   Keep the explanation concise and easy to understand.
                             *   Do not simply repeat the raw solution output; interpret it.
                             """

                             try:
                                 # Add a spinner for the third API call
                                 with st.spinner("💡 Generating interpretation..."):
                                     interpretation_response = client.models.generate_content(
                                         model='gemini-2.0-flash', # Or your preferred model
                                         contents=interpretation_prompt,
                                         # safety_settings=... # Optional safety settings
                                     )
                                     interpretation_text = interpretation_response.text
                                     st.markdown(interpretation_text) # Display the interpretation

                             except Exception as interp_error:
                                 st.error(f"An error occurred while generating the interpretation: {interp_error}")
                             # --- END NEW Interpretation Section ---

                        else:
                             st.info("The generated script ran but did not print any output to standard out, so no interpretation can be provided.")

                    except Exception as exec_error:
                         st.error(f"Error executing the generated Python code:\n```\n{exec_error}\n```")
                         st.error("There might be an issue with the generated code itself.")

            except Exception as code_gen_error:
                st.error(f"An error occurred while generating the Python code: {code_gen_error}")
            # --- END Section to generate/execute Scipy code ---

        else:
            # This warning message is shown if the function returned None
            st.warning("Could not automatically extract the concise formulation section from the response.")
            st.info("Make sure the LLM response includes a section starting with 'Concise LaTeX Formulation:'. The extraction might have failed if the formatting is unexpected.")

    except Exception as e:
        st.error(f"An error occurred while contacting the AI: {e}")
        st.error("Please check your prompt or API key configuration.")


# You can add any Streamlit elements here, just like in your main app.py
# if st.button("Click me on the LLM page!"):
#     st.balloons()
# (Optional: you might want to remove or repurpose this button) 