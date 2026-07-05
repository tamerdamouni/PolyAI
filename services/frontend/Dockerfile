FROM node:20-slim

WORKDIR /app

ARG NEXT_PUBLIC_AGENT_URL=http://localhost:8000
ENV NEXT_PUBLIC_AGENT_URL=$NEXT_PUBLIC_AGENT_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
