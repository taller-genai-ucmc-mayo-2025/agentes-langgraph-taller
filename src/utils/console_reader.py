import sys
from typing import Optional, Dict, AsyncGenerator
from colorama import init, Fore, Style
import re

init()

class ConsoleReader:
    def __init__(
        self,
        fallback: Optional[str] = None,
        input_prompt: str = "User 👤 : ",
        allow_empty: bool = False
    ):
        self.fallback = fallback
        self.input_prompt = input_prompt
        self.allow_empty = allow_empty
        self.is_active = True

    def write(self, role: str, data: str) -> None:
        """Write a message to the console with an optional role prefix."""
        # Remove ANSI codes from data if present
        clean_data = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', data or '')
        parts = []
        if role:
            parts.append(f"{Fore.RED}{Style.BRIGHT}{role}{Style.RESET_ALL}")
        parts.append(clean_data)
        print(" ".join(filter(None, parts)))
        print(data)

    async def prompt(self) -> str:
        """Return the next prompt from the iterator."""
        async for item in self:
            return item["prompt"]
        sys.exit(0)

    async def ask_single_question(self, query_message: str) -> str:
        """Ask a single question and return the answer."""
        colored_query = f"{Fore.CYAN}{Style.BRIGHT}{query_message}{Style.RESET_ALL}"
        answer = input(colored_query)
        # Remove ANSI codes from answer
        clean_answer = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', answer).strip()
        return clean_answer

    def close(self) -> None:
        """Close the reader."""
        self.is_active = False
        # No direct equivalent to pausing stdin in Python; just mark as inactive

    async def __aiter__(self) -> AsyncGenerator[Dict[str, any], None]: # type: ignore
        """Async iterator for reading prompts."""
        if not self.is_active:
            return

        try:
            self.write("", "Interactive session has started. To escape, input 'q' and submit.")

            iteration = 1
            while self.is_active:
                prompt = input(f"{Fore.CYAN}{Style.BRIGHT}{self.input_prompt}{Style.RESET_ALL}")
                # Remove ANSI codes
                prompt = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', prompt).strip()

                if prompt == "q":
                    break

                if not prompt.strip():
                    prompt = self.fallback or ""

                if not self.allow_empty and not prompt.strip():
                    self.write("", "Error: Empty prompt is not allowed. Please try again.")
                    continue

                yield {"prompt": prompt, "iteration": iteration}
                iteration += 1

        except KeyboardInterrupt:
            pass
        finally:
            self.is_active = False
            self.close()