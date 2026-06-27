export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
}

export interface ChatResponse {
  response: string;
  annotated_image?: string | null;
}
