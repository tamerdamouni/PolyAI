export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
  image_url?: string;
}

export interface ChatResponse {
  response: string;
  annotated_image_url?: string | null;
}
