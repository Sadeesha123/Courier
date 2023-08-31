import axios from 'axios';

const url = "http://178.128.24.144"

const url_ml = ""

const api = axios.create({
    baseURL: url,//---http://localhost:8000
});

export const post = async (urlx, data, params = {}, files = []) => {
    try {
                const formData = new FormData();
                files.forEach(file => {
                    formData.append('files', file);
                });

                Object.entries(data).forEach(([key, value]) => {
                    formData.append(key, value);
                });
                const response = await api.post(
                    urlx, 
                    formData, 
                    {   params, 
                        headers: { 
                            'Content-Type': 'multipart/form-data' 
                        } 
                    });
                
                return response.data;
        
    } catch (error) {
        console.log(error)
    }
};