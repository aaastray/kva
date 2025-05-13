// src/services/api.ts
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Убедитесь, что порт совпадает

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  // headers: { // Для POST с JSON будет application/json по умолчанию. Для FormData axios сам установит.
  //   'Content-Type': 'application/json',
  // }
});

apiClient.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error.response || error.message);
    if (error.response && error.response.data && error.response.data.detail) {
        // Если бэкенд возвращает ошибку в формате FastAPI (поле detail)
        if (typeof error.response.data.detail === 'string') {
            return Promise.reject(new Error(error.response.data.detail));
        } else if (Array.isArray(error.response.data.detail) && error.response.data.detail.length > 0) {
            // Если FastAPI возвращает массив ошибок валидации
            const messages = error.response.data.detail.map((err: any) => `${err.loc.join('.')} - ${err.msg}`).join('; ');
            return Promise.reject(new Error(messages || 'Ошибка валидации'));
        }
    }
    return Promise.reject(error.message || new Error('Произошла неизвестная ошибка API'));
  }
);

export default apiClient;

export const getFullImageUrl = (imagePath: string) => {
  if (!imagePath) return '';
  if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
      return imagePath;
  }
  return `${API_BASE_URL}${imagePath}`;
};