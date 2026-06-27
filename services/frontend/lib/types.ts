export interface TokenUsage {
  input: number;
  output: number;
  total: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
  tokens_used?: TokenUsage; // TEMP: visual test of token counting, remove later
}

export interface ChatResponse {
  response: string;
  annotated_image?: string | null;
  tokens_used?: TokenUsage; // TEMP: visual test of token counting, remove later
}
