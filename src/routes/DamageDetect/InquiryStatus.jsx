import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus, faStarHalfStroke } from "@fortawesome/free-solid-svg-icons";
import Sidepanel from "../../components/sidepanel";
import axios from "axios";

function InquiryStatus() {
  const handleSubmit = (event) => {
    event.preventDefault();
  };


  const [images, setImages] = useState([]);

  useEffect(() => {
  
    axios.get("api-endpoint")
      .then(response => {
        
        setImages(response.data);
      })
      .catch(error => {
        console.log("Error fetching image data:", error);
      });
  }, []);



  return (
    <div className="main-body h-screen w-full bg-slate-100">
      <div className="main-body-container h-screen w-full flex flex-row">
        <Sidepanel />
        <div className="w-5/6 side-panel bg-slate-100 p-5">
          <div className="common-body p-5 flex flex-col h-full items-center justify-center">
            <div className="form-body w-[80%] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">
              <form onSubmit={handleSubmit} className="flex flex-col w-full">
                <label
                  htmlFor="orderId"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package Before Delivery
                </label>
                <div className="image-holder flex">
                  {images.map((image, index) => (
                    <img
                      key={index}
                      src={image.url}
                      alt={`Image ${index + 1}`}
                      className="image-box w-[200px] mr-[10px]"
                    />
                  ))}
                </div>

                <label
                  htmlFor="orderId"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package After Delivery
                </label>
                <div className="image-holder flex">
                  {images.map((image, index) => (
                    <img
                      key={index}
                      src={image.url}
                      alt={`Image ${index + 1}`}
                      className="image-box w-[200px] mr-[10px]"
                    />
                  ))}
                </div>

                <span>Package Status</span>

                <div className="w-full text-right ">
                  <Link to="/CustomerHome"><input
                    type="button"
                    value="Next"
                    className="py-2 px-4 bg-blue-500 text-white rounded-lg w-1/4 "
                  /></Link>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default InquiryStatus;
